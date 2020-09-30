import os
import glob
import random
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchtext import data
from utils import *
from classifier_svgd import Classification


parser = ArgumentParser()
parser.add_argument("--corpus",
                    type=str,
                    choices=['yelp', 'snli', 'age'],
                    default='yelp')
parser.add_argument('--epochs',
                    type=int,
                    default=30)
parser.add_argument('--batch_size',
                    type=int,
                    default=128)
parser.add_argument("--activation",
                    type=str,
                    choices=['tanh', 'relu', 'leakyrelu'],
                    default='relu')
parser.add_argument("--optimizer",
                    type=str,
                    choices=['rprop',
                             'adadelta',
                             'adagrad',
                             'rmsprop',
                             'adamax',
                             'asgd',
                             'adam',
                             'sgd'],
                    default='adam')
parser.add_argument('--embed_dim',
                    type=int,
                    default=300)
parser.add_argument('--fc_dim',
                    type=int,
                    default=3000)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=300)
parser.add_argument('--attention_unit',
                    type=int,
                    default=350,
                    help='number of attention unit')
parser.add_argument('--attention_hops',
                    type=int,
                    default=15,
                    help='number of attention hops, for multi-hop attention model')
parser.add_argument('--dropout',
                    type=float,
                    default=0.3)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0005)
parser.add_argument('--lr_patience',
                    type=int,
                    default=1)
parser.add_argument('--lr_decay',
                    type=float,
                    default=0.99)
parser.add_argument('--lr_reduction_factor',
                    type=float,
                    default=0.2)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--preserve_case',
                    action='store_false',
                    dest='lower')
parser.add_argument('--word_embedding',
                    type=str,
                    default='glove.840B.300d')
parser.add_argument('--early_stopping_patience',
                    type=int,
                    default=2)
parser.add_argument('--save_dir_name',
                    type=str,
                    default='tmp')
parser.add_argument('--resume_snapshot',
                    type=str,
                    default='')
parser.add_argument('--seed',
                    type=int,
                    default=1234)
parser.add_argument('--clip',
                    type=float,
                    default=0.5,
                    help='clip to prevent the too large grad in LSTM')
parser.add_argument('--penalization_coeff',
                    type=float,
                    default=0,
                    help='the penalization coefficient')
parser.add_argument('--alpha',
                    type=float,
                    default=1,
                    help='weight of dkernel in svgd and spos')
parser.add_argument('--stepsize',
                    type=float,
                    default=1,
                    help='stepsize in update rule in svgd and spos')
parser.add_argument('--beta',
                    type=float,
                    default=1,
                    help='parameter in spos')
parser.add_argument('--bayesian_method',
                    type=str,
                    choices=['None', 'svgd', 'spos'],
                    default='None')


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_DIR = os.path.join(ROOT_DIR, 'vector_cache/')
DATA_DIR = os.path.join(ROOT_DIR, 'data/')
RESULT_DIR = os.path.join(ROOT_DIR, 'results/')


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def main():
    print('Start')
    # config
    config = parser.parse_args()
    config.save_path = os.path.join(RESULT_DIR, config.save_dir_name)
    make_dirs(config.save_path)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.cuda.device(config.gpu)

    if config.penalization_coeff > 0:
        I = Variable(torch.zeros(config.batch_size, config.attention_hops, config.attention_hops))
        for i in range(config.batch_size):
            for j in range(config.attention_hops):
                I.data[i][j][j] = 1
        I = I.cuda()

    # read data
    print('read corpus data...')
    index_field = data.Field(sequential=False, use_vocab=None)
    label_field = data.Field(sequential=False, unk_token=None)
    input_field = data.Field(lower=True, tokenize='spacy')
    train, dev, test = read_data(config.corpus, input_field, label_field, index_field)
    input_field.build_vocab(train, dev, test)
    label_field.build_vocab(train)
    print('Finish!')

    # set word embedding
    if config.word_embedding:
        pretrained_embedding = os.path.join(VOCAB_DIR + config.corpus + '_' + config.word_embedding + '.pt')
        if os.path.isfile(pretrained_embedding):
            input_field.vocab.vectors = torch.load(pretrained_embedding,
                                                   map_location=lambda storage, location: storage.cuda(config.gpu))
            # input_field.vocab.vectors = torch.load(pretrained_embedding,
            #                                        map_location=lambda storage, location: storage.cuda(0))
            # input_field.vocab.vectors = torch.load(pretrained_embedding, map_location=lambda storage, location: storage)
        else:
            print('Downloading pretrained {} word embeddings\n'.format(config.word_embedding))
            input_field.vocab.load_vectors(config.word_embedding)
            make_dirs(os.path.dirname(pretrained_embedding))
            torch.save(input_field.vocab.vectors, pretrained_embedding)

    # Iterator
    if config.corpus in ('yelp', 'age'):
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                     sort_key=lambda x: len(x.input),
                                                                     batch_size=config.batch_size,
                                                                     device=torch.device('cuda'))
    elif config.corpus == 'snli':
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                     sort_key=lambda x: data.interleave_keys(
                                                                         len(x.premise), len(x.hypothesis)),
                                                                     batch_size=config.batch_size,
                                                                     device=torch.device('cuda'))
    # other config
    config.embed_size = len(input_field.vocab)
    config.out_dim = len(label_field.vocab)

    if config.resume_snapshot:
        model = torch.load(config.resume_snapshot,
                           map_location=lambda storage, location: storage.cuda(config.gpu))
    else:
        model = Classification(config)
        if config.word_embedding:
            model.cuda(device=config.gpu)
            model.sentence_embedding.word_embedding.weight.data = input_field.vocab.vectors
            model.sentence_embedding.word_embedding.weight.requires_grad = True

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config.optimizer == 'adadelta':
        optim_algorithm = optim.Adadelta
    elif config.optimizer == 'adagrad':
        optim_algorithm = optim.Adagrad
    elif config.optimizer == 'adam':
        optim_algorithm = optim.Adam
    elif config.optimizer == 'adamax':
        optim_algorithm = optim.Adamax
    elif config.optimizer == 'asgd':
        optim_algorithm = optim.ASGD
    elif config.optimizer == 'rmsprop':
        optim_algorithm = optim.RMSprop
    elif config.optimizer == 'rprop':
        optim_algorithm = optim.Rprop
    elif config.optimizer == 'sgd':
        optim_algorithm = optim.SGD
    else:
        raise Exception('Unknown optimization optimizer: "%s"' % config.optimizer)

    optimizer = optim_algorithm(model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               'min',
                                               factor=config.lr_reduction_factor,
                                               patience=config.lr_patience,
                                               verbose=False,
                                               min_lr=1e-5)

    iterations = 0
    best_dev_acc = -1
    dev_accuracies = []
    best_dev_loss = 1
    early_stopping = 0
    stop_training = False
    train_iter.repeat = False
    make_dirs(config.save_path)

    # Print parameters and config
    print('Config: ')
    print(config)
    print('\n')

    # Print the model
    print('Model:\n')
    print(model)
    print('\n')
    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))
    print('\nTraining started...\n')

    # Train for the number of epochs specified
    for epoch in range(config.epochs):
        if stop_training:
            break

        train_iter.init_epoch()
        n_correct = 0
        n_total = 0
        train_accuracies = []
        all_losses = []

        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0][
                                              'lr'] * config.lr_decay if epoch > 0 and config.optimizer == 'sgd' else \
            optimizer.param_groups[0]['lr']
        print('\nEpoch: {}/{}'.format(epoch + 1, config.epochs), end=' ')
        print('(Learning rate: {})'.format(optimizer.param_groups[0]['lr']))

        for batch_idx, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()
            iterations += 1

            # forward
            answer, attention = model(batch)

            # Calculate accuracy
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct.cpu().numpy() / n_total
            train_accuracies.append(train_acc)

            # Calculate loss
            loss = criterion(answer, batch.label)
            all_losses.append(loss.item())

            # add penalization term
            if config.penalization_coeff > 0:  # add penalization term
                attentionT = torch.transpose(attention, 1, 2).contiguous()
                extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
                loss += config.penalization_coeff * extra_loss

            # Backpropagate and update the learning rate
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()

            print(
                'Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:6.2f}% - Accuracy(avg): {:6.2f}% - Accuracy: {:6.2f}%'.format(
                    100. * (1 + batch_idx) / len(train_iter),
                    1 + batch_idx, len(train_iter),
                    round(100. * np.mean(all_losses), 2),
                    round(np.mean([x.item() for x in train_accuracies]), 2),
                    round(train_acc, 2)), end='\r')

            # Evaluate performance
            if 1 + batch_idx == len(train_iter):
                # Switch model to evaluation mode
                model.eval()
                dev_iter.init_epoch()

                # Calculate Accuracy
                n_dev_correct = 0
                dev_loss = 0
                dev_losses = []

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    answer, attention = model(dev_batch)
                    n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == \
                                      dev_batch.label.data).sum()
                    dev_loss = criterion(answer, dev_batch.label)
                    dev_losses.append(dev_loss.item())

                dev_acc = 100. * n_dev_correct.cpu().numpy() / len(dev)
                dev_accuracies.append(dev_acc)

                print('\nDev loss: {}% - Dev accuracy: {}%'.format(round(100. * np.mean(dev_losses), 2),
                                                                   round(dev_acc, 2)))

                # Update validation best accuracy if it is better than already stored
                if dev_acc > best_dev_acc:

                    best_dev_acc = dev_acc
                    best_dev_epoch = 1 + epoch
                    snapshot_prefix = os.path.join(config.save_path, 'best')

                    dev_snapshot_path = snapshot_prefix + \
                                        '_{}_{}D_devacc_{}_epoch_{}.pt'.format(config.save_dir_name, config.hidden_dim,
                                                                               dev_acc, 1 + epoch)

                    # save model, delete previous snapshot
                    torch.save(model, dev_snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != dev_snapshot_path:
                            os.remove(f)

                    # Check for early stopping
                    early_stopping = 0
                else:
                    early_stopping += 1

                if early_stopping > config.early_stopping_patience and config.optimizer != 'sgd':
                    stop_training = True
                    print('\nEarly stopping')

                if config.optimizer == 'sgd' and optimizer.param_groups[0]['lr'] < 1e-5:
                    stop_training = True
                    print('\nEarly stopping')

                # Update learning rate
                scheduler.step(round(np.mean(dev_losses), 2))
                dev_losses = []

            # If training has completed, calculate the test scores
            if stop_training == True or (1 + epoch == config.epochs and 1 + batch_idx == len(train_iter)):
                print('\nTraining completed after {} epocs.\n'.format(1 + epoch))

                # Save the final model
                final_snapshot_prefix = os.path.join(config.save_path, 'final')
                final_snapshot_path = final_snapshot_prefix + \
                                      '_{}_{}D.pt'.format(config.save_dir_name, config.hidden_dim)
                torch.save(model, final_snapshot_path)
                for f in glob.glob(final_snapshot_prefix + '*'):
                    if f != final_snapshot_path:
                        os.remove(f)

                # Evaluate the best dev model
                test_model = torch.load(dev_snapshot_path)
                # Switch model to evaluation mode
                test_model.eval()
                test_iter.init_epoch()

                # Calculate Accuracy
                n_test_correct = 0
                test_loss = 0
                test_losses = []

                for test_batch_idx, test_batch in enumerate(test_iter):
                    answer, attention = test_model(test_batch)
                    n_test_correct += (torch.max(answer, 1)[1].view(
                        test_batch.label.size()).data == test_batch.label.data).sum()
                    test_loss = criterion(answer, test_batch.label)
                    test_losses.append(test_loss.item())

                test_acc = 100. * n_test_correct.cpu().numpy() / len(test)
                print('\n')
                print('SUMMARY:\n')
                print('Mean dev accuracy: {:6.2f}%'.format(round(np.mean([x.item() for x in dev_accuracies])), 2))
                print('BEST MODEL:')
                print('Early stopping patience: {}'.format(config.early_stopping_patience))
                print('Epoch: {}'.format(best_dev_epoch))
                # print('Dev accuracy: {:<6.2f}%'.format(round(np.mean(best_dev_acc), 2)))
                print('Dev accuracy: {:<6.2f}%'.format(round(best_dev_acc, 2)))
                print('Test loss: {:<.2f}%'.format(round(100. * np.mean(test_losses), 2)))
                # print('Test accuracy: {:<5.2f}%\n'.format(round(np.mean(test_acc), 2)))
                print('Test accuracy: {:<5.2f}%\n'.format(round(test_acc, 2)))


if __name__ == '__main__':
    main()

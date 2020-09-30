import pickle
import os
import json
import random
import csv
import errno
import torch
import numpy as np
from torchtext.data import TabularDataset


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data/')


def split_yelp(yelp_json_file, train_num, dev_num, test_num):
    print('Load data...')
    with open(os.path.join(DATA_DIR, yelp_json_file)) as f:
        lines_all = f.readlines()
    print('All samples number:', len(lines_all))
    lines = lines_all[:train_num+dev_num+test_num]
    print('Subset samples number:', len(lines))
    print('Shuffle....')
    random.shuffle(lines)
    with open(os.path.join(DATA_DIR, 'review_shuffle.pkl'), 'wb') as f:
        pickle.dump(lines, f)
    print('Finish!')
    stars = []
    text = []

    print('Start processing...')
    for i, line in enumerate(lines):
        item = json.loads(line)
        stars.append(item['stars'])
        text.append(item['text'])

        if i % 100 == 99:
            print('%d/%d files done' %
                  (i + 1, len(lines)))

    stars_train = stars[:train_num]
    text_train = text[:train_num]
    stars_dev = stars[train_num: train_num+dev_num]
    text_dev = text[train_num: train_num+dev_num]
    stars_test = stars[train_num+dev_num: train_num+dev_num+test_num]
    text_test = text[train_num+dev_num: train_num+dev_num+test_num]
    print('Training set:', len(stars_train))
    print('Validation set:', len(stars_dev))
    print('Testing set:', len(stars_test))

    rows_train = zip(stars_train, text_train)
    rows_dev = zip(stars_dev, text_dev)
    rows_test = zip(stars_test, text_test)

    print('Saving...')
    with open(os.path.join(DATA_DIR, 'train_yelp.csv'), "w") as f:
        writer = csv.writer(f)
        for row in rows_train:
            writer.writerow(row)
    with open(os.path.join(DATA_DIR, 'dev_yelp.csv'), "w") as f:
        writer = csv.writer(f)
        for row in rows_dev:
            writer.writerow(row)
    with open(os.path.join(DATA_DIR, 'test_yelp.csv'), "w") as f:
        writer = csv.writer(f)
        for row in rows_test:
            writer.writerow(row)
    print('Finish!')


def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def read_data(corpus, input_field, label_field, index_field):
    path = DATA_DIR
    if corpus == 'yelp':
        train_name = 'train_yelp.csv'
        dev_name = 'dev_yelp.csv'
        test_name = 'test_yelp.csv'
        datafield = [("label", label_field), ("input", input_field)]

    elif corpus == 'snli':
        train_name = 'snli_1.0_train_d.csv'
        dev_name = 'snli_1.0_dev_d.csv'
        test_name = 'snli_1.0_test_d.csv'
        datafield = [("index", index_field), ("label", label_field), ("premise", input_field),
                     ("hypothesis", input_field)]

    elif corpus == 'age':
        train_name = 'age2_train.csv'
        dev_name = 'age2_valid.csv'
        test_name = 'age2_test.csv'
        datafield = [("label", label_field), ("input", input_field)]

    train, dev, test = TabularDataset.splits(
        path=path,
        train=train_name, validation=dev_name, test=test_name,
        format='csv',
        skip_header=None,
        fields=datafield,
        filter_pred=lambda ex: ex.label != '-')

    return train, dev, test


def rbf(theta, h=-1):
    pn = theta.size(0) * torch.ones((1), device='cuda')
    theta_ = theta.unsqueeze(1)
    pdist_square = ((theta - theta_) ** 2).sum(dim=-1)
    if h < 0:
        median = torch.median(pdist_square.view(theta.size(0) ** 2))
        h = torch.sqrt(0.5 * median / torch.log(pn + 1.0))

    kernel = torch.exp(-pdist_square / h ** 2 / 2)
    kernel_sum = torch.sum(kernel, dim=-1, keepdim=True)
    kernel_grad = (-torch.matmul(kernel, theta) + theta * kernel_sum) / h

    return kernel, kernel_grad


def svgd(theta, dtheta, dkernel_weight, h=-1):
    pn = theta.size(0) * torch.ones((1), device='cuda')
    kernel, kernel_grad = rbf(theta, h)
    theta_grad = torch.div(torch.matmul(kernel, dtheta) - dkernel_weight * kernel_grad, pn)

    return theta_grad


def spos(theta, dtheta, dkernel_weight, beta, stepsize, h=-1):
    theta_grad = svgd(theta, dtheta, dkernel_weight, h)
    theta_grad = theta_grad + dtheta * beta - np.sqrt(2 * beta / stepsize) * torch.randn_like(theta_grad) * torch.median(abs(dtheta))
    # theta_grad = theta_grad + dtheta * beta - np.sqrt(2 * beta / stepsize) * torch.randn_like(theta_grad)
    # theta_grad = theta_grad * beta - np.sqrt(2 * beta / stepsize) * torch.randn_like(theta_grad)
    return theta_grad


if __name__ == '__main__':
    split_yelp('review.json', 500000, 2000, 2000)










































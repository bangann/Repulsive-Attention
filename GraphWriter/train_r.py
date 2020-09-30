import sys
from random import shuffle
import os
from math import exp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lastDataset import dataset
from pargs import pargs, dynArgs
from models.newmodel import model
from sklearn.metrics.pairwise import cosine_similarity


def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
        o.param_groups[0]['lr'] = args.lrhigh
    else:
        o.param_groups[0]['lr'] -= args.lrchange


def train(m, o, ds, args):
    print("Training", end="\t")
    loss = 0
    ex = 0
    trainorder = [('1', ds.t1_iter), ('2', ds.t2_iter), ('3', ds.t3_iter)]
    # trainorder = reversed(trainorder)
    shuffle(trainorder)
    for spl, train_iter in trainorder:
        print(spl)
        for count, b in enumerate(train_iter):
            if count % 100 == 99:
                print(ex, "of like 40k -- current avg loss ", (loss / ex))
            b = ds.fixBatch(b)
            p, z, planlogits = m(b)
            p = p[:, :-1, :].contiguous()

            tgt = b.tgt[:, 1:].contiguous().view(-1).to(args.device)
            l = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
            # copy coverage (each elt at least once)
            if args.cl:
                z = z.max(1)[0]
                cl = nn.functional.mse_loss(z, torch.ones_like(z))
                l = l + args.cl * cl
            if args.plan:
                pl = nn.functional.cross_entropy(planlogits.view(-1, planlogits.size(2)), b.sordertgt[0].view(-1),
                                                 ignore_index=1)
                l = l + args.plweight * pl

            k_w = m.attn.key_layer.weight.clone().detach()
            k_w = k_w.t().contiguous().view(args.heads, -1)
            k_dist = np.mean(cosine_similarity(k_w.cpu().detach().numpy(), k_w.cpu().detach().numpy()))

            q_w = m.attn.query_layer.weight.clone().detach()
            q_w = q_w.t().contiguous().view(args.heads, -1)
            q_dist = np.mean(cosine_similarity(q_w.cpu().detach().numpy(), q_w.cpu().detach().numpy()))

            v_w = m.attn.value_layer.weight.clone().detach()
            v_w = v_w.t().contiguous().view(args.heads, -1)
            v_dist = np.mean(cosine_similarity(v_w.cpu().detach().numpy(), v_w.cpu().detach().numpy()))

            l = l + 0.01 * (k_dist + q_dist + v_dist)

            l.backward()

            # if args.bayesian_method != 'None':
            #     theta = m.attn.key_layer.weight.clone().detach().requires_grad_(False)
            #     dtheta = m.attn.key_layer.weight.grad.data.clone()
            #     current_grad = weight_bayesian(theta, dtheta, args.heads)
            #     m.attn.key_layer.weight.grad.data = current_grad.cuda()
            #
            #     theta = m.attn.query_layer.weight.clone().detach().requires_grad_(False)
            #     dtheta = m.attn.query_layer.weight.grad.data.clone()
            #     current_grad = weight_bayesian(theta, dtheta, args.heads)
            #     m.attn.query_layer.weight.grad.data = current_grad.cuda()
            #
            #     theta = m.attn.value_layer.weight.clone().detach().requires_grad_(False)
            #     dtheta = m.attn.value_layer.weight.grad.data.clone()
            #     current_grad = weight_bayesian(theta, dtheta, args.heads)
            #     m.attn.value_layer.weight.grad.data = current_grad.cuda()






            nn.utils.clip_grad_norm_(m.parameters(), args.clip)
            loss += l.item() * len(b.tgt)
            o.step()
            o.zero_grad()
            ex += len(b.tgt)
    loss = loss / ex
    print("AVG TRAIN LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))


def evaluate(m, ds, args):
    print("Evaluating", end="\t")
    m.eval()
    loss = 0
    ex = 0
    for b in ds.val_iter:
        b = ds.fixBatch(b)
        p, z, planlogits = m(b)
        p = p[:, :-1, :]
        tgt = b.tgt[:, 1:].contiguous().view(-1).to(args.device)
        l = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
        if ex == 0:
            g = p[0].max(1)[1]
            print(ds.reverse(g, b.rawent[0]))
        loss += l.item() * len(b.tgt)
        ex += len(b.tgt)
    loss = loss / ex
    print("VAL LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))
    m.train()
    return loss


def main(args):
    try:
        os.stat(args.save)
        input("Save File Exists, OverWrite? <CTL-C> for no")
    except:
        os.mkdir(args.save)
    ds = dataset(args)
    args = dynArgs(args, ds)
    m = model(args)
    print(args.device)
    m = m.to(args.device)
    if args.ckpt:
        '''
        with open(args.save+"/commandLineArgs.txt") as f:
          clargs = f.read().strip().split("\n") 
          argdif =[x for x in sys.argv[1:] if x not in clargs]
          assert(len(argdif)==2); 
          assert([x for x in argdif if x[0]=='-']==['-ckpt'])
        '''
        cpt = torch.load(args.ckpt)
        m.load_state_dict(cpt)
        starte = int(args.ckpt.split("/")[-1].split(".")[0]) + 1
        args.lr = float(args.ckpt.split("-")[-1])
        print('ckpt restored')
    else:
        with open(args.save + "/commandLineArgs.txt", 'w') as f:
            f.write("\n".join(sys.argv[1:]))
        starte = 0
    o = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)

    # early stopping based on Val Loss
    lastloss = 1000000

    for e in range(starte, args.epochs):
        print("epoch ", e, "lr", o.param_groups[0]['lr'])
        train(m, o, ds, args)
        vloss = evaluate(m, ds, args)
        if args.lrwarm:
            update_lr(o, args, e)
        print("Saving model")
        torch.save(m.state_dict(),
                   args.save + "/" + str(e) + ".vloss-" + str(vloss)[:8] + ".lr-" + str(o.param_groups[0]['lr']))
        if vloss > lastloss:
            if args.lrdecay:
                print("decay lr")
                o.param_groups[0]['lr'] *= 0.5
        lastloss = vloss


def svgd(theta, dtheta, dkernel_weight, h=-1):
    pn = theta.size(0) * torch.ones((1), device='cuda')
    kernel, kernel_grad = rbf(theta, h)
    # dtheta = theta / 60000 + dtheta
    theta_grad = torch.div(torch.matmul(kernel, dtheta) - dkernel_weight * kernel_grad, pn)

    return theta_grad


def spos(theta, dtheta, dkernel_weight, beta, stepsize, h=-1):
    theta_grad = svgd(theta, dtheta, dkernel_weight, h)
    theta_grad = theta_grad + dtheta * beta - np.sqrt(2.0 * beta / stepsize) * torch.randn_like(theta_grad)
    # theta_grad = theta_grad * beta - np.sqrt(2 * beta / stepsize) * torch.randn_like(theta_grad)
    return theta_grad


def rbf(theta, h=-1):
    pn = theta.size(0) * torch.ones((1), device='cuda')
    theta_ = theta.unsqueeze(1)
    pdist_square = ((theta - theta_) ** 2).sum(dim=-1)
    if h < 0:
        median = torch.median(pdist_square.view(theta.size(0) ** 2))
        h = torch.sqrt(0.5 * median / torch.log(pn + 1.0))

    kernel = torch.exp(-pdist_square / h ** 2 / 2.0)
    kernel_sum = torch.sum(kernel, dim=-1, keepdim=True)
    kernel_grad = (-torch.matmul(kernel, theta) + theta * kernel_sum) / h

    return kernel, kernel_grad


def weight_bayesian(theta, dtheta_data, heads_num):
    theta_t = theta.t().contiguous().view(heads_num, -1)

    dtheta_data_t = dtheta_data.t().contiguous().view(heads_num, -1)

    if args.bayesian_method == 'svgd':
        current_grad = svgd(theta_t, dtheta_data_t, args.d_kernel_weight)

    elif args.bayesian_method == 'spos':
        current_grad = spos(theta_t, dtheta_data_t, args.d_kernel_weight, args.beta, args.stepsize)

    current_grad = current_grad.view(theta.size()[1], theta.size()[0]).t()

    return current_grad


if __name__ == "__main__":
    args = pargs()
    main(args)

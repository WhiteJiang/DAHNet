import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate

from tqdm import tqdm
from loguru import logger
from models.ADSH_Loss import ADSH_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.DAHNET as DAHNET
import torch.nn as nn
import time


def train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, feat_size = args.num_classes, 2048

    model = DAHNET.dahnet(code_length=code_length, num_classes=num_classes, feat_size=feat_size,
                          device=args.device, pretrained=True)

    model = model.cuda()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=True)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)
    cross = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 50], gamma=0.1)
    # loss
    criterion = ADSH_Loss(code_length, args.gamma)

    num_retrieval = len(retrieval_dataloader.dataset)  # len = train data

    U = torch.zeros(args.num_samples, code_length).cuda()
    B = torch.randn(num_retrieval, code_length).cuda()
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().cuda()  # len = train data
    # print(len(retrieval_targets))
    cnn_losses, hash_losses, quan_losses = AverageMeter(), AverageMeter(), AverageMeter()
    cross_loss = AverageMeter()
    start = time.time()
    best_mAP = 0
    for it in range(args.max_iter):
        cur_lr = scheduler.get_lr()
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size,
                                                           args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().cuda()  # len = num samples
        S = (train_targets @ retrieval_targets.t() > 0).float()  # num samples * train num
        # print(S[:1])
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        # print(r)
        S = S * (1 + r) - r
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()
            cross_loss.reset()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
                data, targets, index = data.cuda(), targets.cuda(), index.cuda()

                optimizer.zero_grad()

                F, local_f, cls, cls1, cls2, cls3 = model(data)
                U[index, :] = F.data
                cnn_loss, hash_loss, quan_loss = criterion(F, B, S[index, :], sample_index[index])

                cls_loss = (1.0 / 2.0) * cross(cls, targets) + \
                           (1.0 / 6.0) * (cross(cls1, targets) + cross(cls2, targets) + cross(cls3, targets))
                cnn_loss = cnn_loss + cls_loss
                cnn_losses.update(cnn_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())
                cross_loss.update(cls_loss.item())
                cnn_loss.backward()
                optimizer.step()
            logger.info(
                '[epoch:{}/{}][cnn_loss:{:.6f}][hash_loss:{:.6f}][quan_loss:{:.6f}][cls_loss:{:.6f}][lr:{}'.format(
                    epoch + 1, args.max_epoch,
                    cnn_losses.avg,
                    hash_losses.avg,
                    quan_losses.avg,
                    cross_loss.avg,
                    cur_lr))
        scheduler.step()
        # Update B
        expand_U = torch.zeros(B.shape).cuda()
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, args.gamma)

        logger.info(
            '[iter:{}/{}][iter_time:{:.2f}][lr:{}]'.format(it + 1, args.max_iter, time.time() - iter_start, cur_lr))

        if it % 1 == 0:
            query_code = generate_code(model, query_dataloader, code_length, args.device)
            # print(len(query_dataloader))
            mAP = evaluate.mean_average_precision(
                query_code.cuda(),
                B,
                query_dataloader.dataset.get_onehot_targets().cuda(),
                retrieval_targets,
                args.device,
                args.topk,
            )
            logger.info(
                '[iter:{}/{}][code_length:{}][mAP:{:.5f}]'.format(it + 1, args.max_iter, code_length,
                                                                  mAP))
            if mAP > best_mAP:
                best_mAP = mAP
                ret_path = os.path.join('checkpoints', args.info, str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.pth'))
                torch.save(B.cpu(), os.path.join(ret_path, 'database_code.pth'))
                torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(ret_path, 'query_targets.pth'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.pth'))
                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pth'))

            logger.info(
                '[iter:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(it + 1, args.max_iter, code_length,
                                                                                   mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time() - start))

    return best_mAP


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).cuda()
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.cuda(), targets.cuda(), index.cuda()
            hash_code, _ = model(data)
            code[index, :] = hash_code.sign()
    model.train()
    return code

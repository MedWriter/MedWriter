import argparse
import logging
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from evaluate import evaluate
from model.sentsat_hia_retrival_cls import REncoder2Decoder_hia
from utils.dataloader import Biview_MultiSent, \
    sent_collate_fn, Openi_Sentence_Retrivel, \
    Openi_OneSentence_Retrivel
from utils.tools import decode, print_example

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='MedWriter')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--report_checkpoint_path', type=str, default='')
    parser.add_argument('--sent_checkpoint_path', type=str, default='')
    parser.add_argument('--dataset-dir', type=str, default='./data')
    parser.add_argument('--train-folds', type=str, default='012')
    parser.add_argument('--val-folds', type=str, default='3')
    parser.add_argument('--test-folds', type=str, default='4')
    parser.add_argument('--report-path', type=str, default='./data/reports.json')
    parser.add_argument('--image-path', type=str, default='../data/imgs')
    parser.add_argument('--vocab-path', type=str, default='./data/openi_vocab.pkl')
    parser.add_argument('--label-path', type=str, default='./data/label_dict.json')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--log-freq', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--decoder-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--gpus', default=[0,1,2,3])
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--clip-value', type=float, default=5.0)

    args = parser.parse_args()
    args.model_dir = os.path.join(args.model_dir, args.name)
    return args


if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.log_dir, args.name + '.log'), level=logging.INFO)
    print('------------------------Model and Training Details--------------------------')
    print(args)
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))

    writer = SummaryWriter(log_dir=os.path.join('./runs', args.name))

    gpus = [int(_) for _ in list(args.gpus)]
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_set = Biview_MultiSent('train',
                                 args.dataset_dir,
                                 args.image_path,
                                 args.train_folds,
                                 args.report_path,
                                 args.vocab_path,
                                 args.label_path,train=True)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              collate_fn=sent_collate_fn)
    val_set = Biview_MultiSent('val',
                               args.dataset_dir,
                               args.image_path,
                               args.val_folds,
                               args.report_path,
                               args.vocab_path,
                               args.label_path,train=False)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            collate_fn=sent_collate_fn)
    test_set = Biview_MultiSent('test',
                                args.dataset_dir,
                                args.image_path,
                                args.test_folds,
                                args.report_path,
                                args.vocab_path,
                                args.label_path,train=False)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True,
                             collate_fn=sent_collate_fn)

    report_retrival = Openi_OneSentence_Retrivel(vocab,
                                               args.train_folds,
                                               args.dataset_dir,
                                               args.report_path,
                                               max_sent_lens=128)
    report_retrival_loader = DataLoader(report_retrival,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True)

    sent_retrival = Openi_Sentence_Retrivel(vocab,
                                                      args.train_folds,
                                                      args.dataset_dir,
                                                      args.report_path,
                                                      max_sent_lens=32)
    sent_retrival_loader = DataLoader(sent_retrival,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=True)



    model = REncoder2Decoder_hia(num_classes=args.num_classes, vocab_size=len(vocab))
    if os.path.isfile(args.report_checkpoint_path):
        print("=> loading checkpoint report retrieval'{}'".format(args.report_checkpoint_path))
        if torch.cuda.is_available():
            checkpoint = torch.load(args.report_checkpoint_path)
        else:
            checkpoint = torch.load(args.report_checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        model.report_r.load_state_dict(checkpoint)
        print("=> loaded checkpoint report retrieval'{}'".format(args.report_checkpoint_path))
    for param in model.report_r.parameters():
        param.requires_grad = False
    model.report_r.eval()

    if os.path.isfile(args.sent_checkpoint_path):
        print("=> loading checkpoint sent retrieval'{}'".format(args.sent_checkpoint_path))
        if torch.cuda.is_available():
            checkpoint = torch.load(args.sent_checkpoint_path)
        else:
            checkpoint = torch.load(args.sent_checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.decoder.sent_retrieval.load_state_dict(checkpoint)
        print("=> loaded checkpoint sent retrieval'{}'".format(args.sent_checkpoint_path))
    for param in model.decoder.sent_retrieval.parameters():
        param.requires_grad = False
    model.decoder.sent_retrieval.eval()

    # if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus).to(device)

    model.module.decoder.sent_retrieval.get_embedding_tensor(sent_retrival_loader, device=device)
    model.module.report_r.get_embedding_tensor(report_retrival_loader, device=device)
    # for param in model.module.encoder.parameters():
    #     param.requires_grad = False
    # model.module.encoder.eval()

    CELoss = nn.CrossEntropyLoss(reduction='none')


    optimizer = torch.optim.Adam([
        # {'params': model.module.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.module.decoder.parameters(), 'lr': args.decoder_lr},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)

    if args.pretrained:
        print('Loading pretrained model from {}'.format(args.pretrained))
        pretrained = torch.load(args.pretrained)
        model.module.encoder.load_state_dict(pretrained['model_state_dict'])

    start_epoch = 1
    best_metric = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']

    num_steps = math.ceil(len(train_set) / args.batch_size)

    val_gts = {}
    test_gts = {}

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.module.decoder.train()
        # model.module.encoder.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        print('------------------------Training for Epoch {}---------------------------'.format(epoch))
        print('Learning rate {:.7f}'.format(optimizer.param_groups[0]['lr']))

        for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(tqdm(train_loader)):
            images1 = images1.to(device)
            images2 = images2.to(device)
            captions = captions.to(device)
            loss_masks = loss_masks.to(device).bool()
            update_masks = update_masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, cls_pred = model(images1, images2, captions[:, :, :-1], update_masks)
            logits = logits.permute(0, 3, 1, 2).contiguous()
            captions = captions[:, :, 1:].contiguous()
            loss_masks = loss_masks[:, :, 1:].contiguous()
            loss_masks = loss_masks.contiguous()
            loss = CELoss(logits, captions)
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred, labels)
            loss = loss.masked_select(loss_masks).mean()
            loss.backward()
            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            clip_grad_value_(model.module.parameters(), args.clip_value)
            optimizer.step()

        epoch_loss /= num_steps
        epoch_cls_loss /= num_steps
        print('Epoch {}/{}, Loss {:.4f}, Cls Loss {:.4f}'.format(epoch, args.num_epochs, epoch_loss, epoch_cls_loss))
        writer.add_scalar('loss', epoch_loss, epoch)

        scheduler.step(epoch)

        if epoch % args.log_freq == 0:

            model.module.decoder.eval()
            # model.module.encoder.eval()

            val_res = {}
            val_gts = {}
            y_true = []
            y_score = []
            for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(tqdm(val_loader)):
                images1 = images1.to(device)
                images2 = images2.to(device)
                preds, cls_pred = model(images1 = images1,
                              images2 = images2,
                              stop_id = vocab('.'))
                val_res.update(decode(vocab, preds, caseids))

                val_gts.update(decode(vocab, captions[:,:,1:], caseids))
                scores = torch.sigmoid(cls_pred).detach().cpu()
                y_true.append(labels.numpy())
                y_score.append(scores.numpy())
            y_score = np.concatenate(y_score, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            y_hat = (y_score >= 0.5).astype(np.int)
            precision, recall, f, _ = precision_recall_fscore_support(y_true, y_hat)
            roc_auc = roc_auc_score(y_true, y_score, average=None)
            print('VAL Epoch {}/{}, P {:.4f}, R {:.4f}, F {:.4f}, AUC {:.4f}'.format(
                epoch, args.num_epochs, precision.mean(), recall.mean(), f.mean(), roc_auc.mean()))

            scores = evaluate(val_gts, val_res)
            print_example(val_gts, val_res, num=2)

            print('VAL BLEU 1 {:3f} BLEU 2 {:3f} BLEU 3 {:3f} BLEU 4 {:3f} ROUGE_L {:3f} CIDEr {:3f}'.format(scores['Bleu_1'],
                                                                                                             scores['Bleu_2'],
                                                                                                             scores['Bleu_3'],
                                                                                                             scores['Bleu_4'],
                                                                                                             scores['ROUGE_L'],
                                                                                                             scores['CIDEr']))

            writer.add_scalar('VAL BLEU 1', scores['Bleu_1'], epoch)
            writer.add_scalar('VAL BLEU 2', scores['Bleu_2'], epoch)
            writer.add_scalar('VAL BLEU 3', scores['Bleu_3'], epoch)
            writer.add_scalar('VAL BLEU 4', scores['Bleu_4'], epoch)
            writer.add_scalar('VAL ROUGE_L', scores['ROUGE_L'], epoch)
            writer.add_scalar('VAL CIDEr', scores['CIDEr'], epoch)
            # writer.add_scalar('VAL Meteor', scores['METEOR'], epoch)
            save_fname = os.path.join(args.model_dir, 'checkpoint.pth'.format(args.name, epoch))
            if len(gpus) > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_fname)
            metric = sum([value for key, value in scores.items()])
            if best_metric <= metric:
                best_metric = metric
                print(20*'*','best model saved',20*'*')
                save_fname = os.path.join(args.model_dir, 'best_model.pth')
                if len(gpus) > 1:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_fname)
            test_res = {}
            test_gts = {}
            y_true = []
            y_score = []
            for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(
                    tqdm(test_loader)):
                images1 = images1.to(device)
                images2 = images2.to(device)
                preds, cls_pred = model(images1=images1,
                              images2=images2,
                              stop_id=vocab('.'))
                test_res.update(decode(vocab, preds, caseids))

                test_gts.update(decode(vocab, captions[:, :, 1:], caseids))
                scores = torch.sigmoid(cls_pred).detach().cpu()
                y_true.append(labels.numpy())
                y_score.append(scores.numpy())
            y_score = np.concatenate(y_score, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            y_hat = (y_score >= 0.5).astype(np.int)
            precision, recall, f, _ = precision_recall_fscore_support(y_true, y_hat)
            roc_auc = roc_auc_score(y_true, y_score, average=None)
            print('Test Epoch {}/{}, P {:.4f}, R {:.4f}, F {:.4f}, AUC {:.4f}'.format(
                epoch, args.num_epochs, precision.mean(), recall.mean(), f.mean(), roc_auc.mean()))

            scores = evaluate(test_gts, test_res)
            print_example(test_gts, test_res, num=2)
            print('TEST BLEU 1 {:3f} BLEU 2 {:3f} BLEU 3 {:3f} BLEU 4 {:3f} ROUGE_L {:3f} CIDEr {:3f}'.format(
                scores['Bleu_1'],
                scores['Bleu_2'],
                scores['Bleu_3'],
                scores['Bleu_4'],
                scores['ROUGE_L'],
                scores['CIDEr']))

            writer.add_scalar('TEST BLEU 1', scores['Bleu_1'], epoch)
            writer.add_scalar('TEST BLEU 2', scores['Bleu_2'], epoch)
            writer.add_scalar('TEST BLEU 3', scores['Bleu_3'], epoch)
            writer.add_scalar('TEST BLEU 4', scores['Bleu_4'], epoch)
            writer.add_scalar('TEST ROUGE_L', scores['ROUGE_L'], epoch)
            writer.add_scalar('TEST CIDEr', scores['CIDEr'], epoch)
            # writer.add_scalar('TEST Meteor', scores['METEOR'], epoch)

            with open(os.path.join(args.output_dir, '{}_test_e{}.csv'.format(args.name, epoch)), 'w') as f1:
                with open(os.path.join(args.output_dir, '{}_test_gts.csv'.format(args.name)), 'w') as f2:
                    for caseid in test_res.keys():
                        f1.write(test_res[caseid][0] + '\n')
                        f2.write(test_gts[caseid][0] + '\n')

            writer.close()
    save_fname = os.path.join(args.model_dir, 'best_model.pth')
    print('Loading best model from {}'.format(args.pretrained))
    pretrained = torch.load(save_fname)
    model.module.load_state_dict(pretrained['model_state_dict'])
    test_res = {}
    test_gts = {}
    y_true = []
    y_score = []
    for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(tqdm(test_loader)):
        images1 = images1.to(device)
        images2 = images2.to(device)
        preds, cls_pred = model(images1 = images1,
                      images2 = images2,
                      stop_id = vocab('.'))
        test_res.update(decode(vocab, preds, caseids))

        test_gts.update(decode(vocab, captions[:,:,1:], caseids))

        scores = torch.sigmoid(cls_pred).detach().cpu()
        y_true.append(labels.numpy())
        y_score.append(scores.numpy())
    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_hat = (y_score >= 0.5).astype(np.int)
    precision, recall, f, _ = precision_recall_fscore_support(y_true, y_hat)
    roc_auc = roc_auc_score(y_true, y_score, average=None)
    print('Test P {:.4f}, R {:.4f}, F {:.4f}, AUC {:.4f}'.format(args.num_epochs, precision.mean(), recall.mean(), f.mean(), roc_auc.mean()))


    scores = evaluate(test_gts, test_res)
    print_example(test_gts, test_res, num=2)
    print('TEST BLEU 1 {:3f} BLEU 2 {:3f} BLEU 3 {:3f} BLEU 4 {:3f} ROUGE_L {:3f} CIDEr {:3f}'.format(
        scores['Bleu_1'],
        scores['Bleu_2'],
        scores['Bleu_3'],
        scores['Bleu_4'],
        scores['ROUGE_L'],
        scores['CIDEr']))

    writer.add_scalar('TEST BLEU 1', scores['Bleu_1'], epoch)
    writer.add_scalar('TEST BLEU 2', scores['Bleu_2'], epoch)
    writer.add_scalar('TEST BLEU 3', scores['Bleu_3'], epoch)
    writer.add_scalar('TEST BLEU 4', scores['Bleu_4'], epoch)
    writer.add_scalar('TEST ROUGE_L', scores['ROUGE_L'], epoch)
    writer.add_scalar('TEST CIDEr', scores['CIDEr'], epoch)
    # writer.add_scalar('TEST Meteor', scores['METEOR'], epoch)

    with open(os.path.join(args.output_dir, '{}_test_best.csv'.format(args.name, epoch)), 'w') as f1:
        with open(os.path.join(args.output_dir, '{}_test_gts.csv'.format(args.name)), 'w') as f2:
            for caseid in test_res.keys():
                f1.write(test_res[caseid][0] + '\n')
                f2.write(test_gts[caseid][0] + '\n')

    writer.close()

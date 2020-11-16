from model.mlclassifier import BertForSequenceClassification
import argparse
from utils import Openi_Report_classification, Openi_Report_Setence_Match
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args,LOSS_FUNC, device):
    model.train()
    losses = AverageMeter('Loss', ':6.3f')

    for index, (input, mask, target) in enumerate(tqdm(train_loader)):
        input, mask, target = input.to(device), mask.to(device), target.to(device)
        output = model(input, attention_mask=mask)
        optimizer.zero_grad()
        loss = LOSS_FUNC(output[0], target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))


        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d]\tIter [%d/%d]\tAvg Loss: %.4f\tLoss: %.4f'
                       % (epoch + 1, args.end_epoch, index + 1, len(train_loader), losses.avg, loss.item()))
    return losses.avg

def test(model, test_loader, nb_classes, LOSS_FUNC, device):
    losses = AverageMeter('Loss', ':6.3f')
    model.eval()
    y_scores = []
    y_true = []
    with torch.no_grad():
        for index, (input, mask, target) in enumerate(tqdm(test_loader)):
            input, mask, target = input.to(device), mask.to(device), target.to(device)
            output = model(input, attention_mask=mask)

            loss = LOSS_FUNC(output[0], target)

            losses.update(loss.item(), input.size(0))

            y_score = torch.sigmoid(output[0])
            y_scores.append(y_score.cpu().numpy())
            y_true.append(target.long().cpu().numpy())

    y_scores = np.concatenate(y_scores,axis=0)
    y_true = np.concatenate(y_true,axis=0)
    aucs = auc(y_scores,y_true,nb_classes)

    print('Mean AUC {}\tVal Loss {losses.avg:.3f}'.format(np.mean(aucs),losses=losses))
    return losses.avg, aucs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def auc(y_scores, y_true, nb_class):
    '''Return a list of AUC for each class'''

    aucs = []
    for c in range(nb_class):
        AUC = roc_auc_score(y_true[:,c], y_scores[:,c])
        aucs.append(AUC)
    return aucs

def main(args):
    if os.path.exists(args.checkpoint_path) == False:
        os.makedirs(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_set = Openi_Report_classification(tokenizer, max_length=args.max_length, report_csv=args.train_csv)
    val_set = Openi_Report_classification(tokenizer, max_length=args.max_length, report_csv=args.val_csv)
    test_set = Openi_Report_classification(tokenizer, max_length=args.max_length, report_csv=args.test_csv)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_class=args.num_class).to(device)
    model.train()
    print(model)

    Loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    metric = []
    for epoch in range(args.start_epoch, args.end_epoch):
        train_loss = train(model,train_loader,optimizer,args.PRINT_INTERVAL,epoch,args, Loss_func, device)
        val_loss, aucs = test(model, val_loader, args.num_class, Loss_func, device)
        mean_aucs = np.mean(aucs)
        scheduler.step()
        metric.append(mean_aucs)
        if max(metric) == mean_aucs:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pth.tar'))
            print("Model Saved")
    print("---------------Testing-----------------")
    test_loss, aucs = test(model, test_loader, args.num_class, Loss_func, device)
    print(aucs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT for report classification')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='an integer for the accumulator')
    parser.add_argument('--lr',type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--train-csv', type=str,
                        default='data/openi_report_train.csv')
    parser.add_argument('--val-csv', type=str,
                        default='data/openi_report_val.csv')
    parser.add_argument('--test-csv', type=str,
                        default='data/openi_report_test.csv')
    parser.add_argument('--end-epoch', type= int, default=50)
    parser.add_argument('--start-epoch', type = int, default=0)
    parser.add_argument('--num-class', type = int, default=50)
    parser.add_argument('--PRINT-INTERVAL', type=int, default=10,
                        help='Number of batch to print the loss')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint/Bert_openi/')
    parser.add_argument('--max-length', type=int, default=128)
    args = parser.parse_args()
    main(args)




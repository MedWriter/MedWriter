from model.LLR import BertForSequenceMatch
import argparse
from utils.dataloader import Openi_Report_classification, Openi_Report_Setence_Match, Openi_Report_Report_Match
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

    for index, (input1, input2, mask1, mask2, target) in enumerate(tqdm(train_loader)):
        input1, mask1,input2, mask2, target = input1.to(device), mask1.to(device),input2.to(device), mask2.to(device), target.to(device)
        output = model(input_ids1=input1, attention_mask1=mask1,
                       input_ids2=input2, attention_mask2=mask2)
        optimizer.zero_grad()
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input1.size(0))


        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d]\tIter [%d/%d]\tAvg Loss: %.4f\tLoss: %.4f'
                       % (epoch + 1, args.end_epoch, index + 1, len(train_loader), losses.avg, loss.item()))
    return losses.avg

def test(model, test_loader, nb_classes, LOSS_FUNC, device):
    losses = AverageMeter('Loss', ':6.3f')
    accs = AverageMeter('Accuracy', ':6.3f')
    model.eval()

    with torch.no_grad():
        for index, (input1, input2, mask1, mask2, target) in enumerate(tqdm(test_loader)):
            input1, mask1, input2, mask2, target = input1.to(device), mask1.to(device), input2.to(device), mask2.to(
                device), target.to(device)
            output = model(input_ids1=input1, attention_mask1=mask1,
                           input_ids2=input2, attention_mask2=mask2)

            loss = LOSS_FUNC(output, target)

            losses.update(loss.item(), input1.size(0))
            output = torch.sigmoid(output)
            output[output > 0.5] = 1
            output[output <= 0.5] = 0

            total = output.shape[0]
            correct = torch.sum(output == target).item()
            acc = correct/total
            accs.update(acc, input1.size(0))

    print('Mean Accuracy {accs.avg:.3f}\tVal Loss {losses.avg:.3f}'.format(accs=accs,losses=losses))
    return losses.avg, accs.avg

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

def main_match(args):
    if os.path.exists(args.checkpoint_path) == False:
        os.makedirs(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_set = Openi_Report_Report_Match(tokenizer, max_length=args.max_length, report_csv=args.train_csv)
    val_set = Openi_Report_Report_Match(tokenizer, max_length=args.max_length, report_csv=args.val_csv)
    test_set = Openi_Report_Report_Match(tokenizer, max_length=args.max_length, report_csv=args.test_csv)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = BertForSequenceMatch.from_pretrained('bert-base-uncased').to(device)
    model.train()
    print(model)

    Loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    metric = []
    for epoch in range(args.start_epoch, args.end_epoch):
        train_loss = train(model,train_loader,optimizer,args.PRINT_INTERVAL,epoch,args, Loss_func, device)
        val_loss, acc = test(model, val_loader, args.num_class, Loss_func, device)

        scheduler.step()
        metric.append(acc)
        if max(metric) == acc:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pth.tar'))
            print("Model Saved")
    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("{}Model Loaded{}".format(20*'-',20*'-'))
    print("{}Testing{}".format(20*'-',20*'-'))
    test_loss, acc = test(model, test_loader, args.num_class, Loss_func, device)
    print('Mean Accuracy {acc:.3f}'.format(acc = acc))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT for report classification')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='an integer for the accumulator')
    parser.add_argument('--lr',type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--train-csv', type=str,
                        default='data/openi_report_train.csv')
    parser.add_argument('--val-csv', type=str,
                        default='data/openi_report_val.csv')
    parser.add_argument('--test-csv', type=str,
                        default='data/openi_report_test.csv')
    parser.add_argument('--end-epoch', type= int, default=100)
    parser.add_argument('--start-epoch', type = int, default=0)
    parser.add_argument('--num-class', type = int, default=50)
    parser.add_argument('--PRINT-INTERVAL', type=int, default=20,
                        help='Number of batch to print the loss')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint/Bert_openi_match/')
    parser.add_argument('--max-length', type=int, default=32)
    args = parser.parse_args()
    main_match(args)




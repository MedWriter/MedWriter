import os
import random
import json
import pickle
import string
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from nltk import sent_tokenize
from utils.tools import Equalize
class Biview_Classification(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, folds, label_fname, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path))
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path))
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        return image1, image2, label

    def get_class_weights(self):
        all_labels = [v for k, v in self.label_dict.items()]
        all_labels = torch.tensor(all_labels, dtype=torch.int)
        num_cases, num_classes = all_labels.size()
        pos_counts = torch.sum(all_labels, dim=0)
        neg_counts = num_cases - pos_counts
        ratio = neg_counts.type(torch.float) / pos_counts.type(torch.float)
        return ratio


class MIMIC_Biview_Classification(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, txt, label_fname, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        with open(txt) as f:
            self.case_list += f.read().splitlines()
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path)).convert('RGB')
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path)).convert('RGB')
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        return image1, image2, label

    def get_class_weights(self):
        all_labels = [v for k, v in self.label_dict.items()]
        all_labels = torch.tensor(all_labels, dtype=torch.int)
        num_cases, num_classes = all_labels.size()
        pos_counts = torch.sum(all_labels, dim=0)
        neg_counts = num_cases - pos_counts
        ratio = neg_counts.type(torch.float) / pos_counts.type(torch.float)
        return ratio

class Biview_OneSent(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, folds, report_fname, vocab_fname, label_fname):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        self.transform = transforms.Compose([
            transforms.RandomCrop((512, 512), pad_if_needed=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path))
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path))
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        report = self.reports[caseid]
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']
        text = text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
        text = text.replace('.', ' .')
        tokens = text.strip().split()
        caption = [self.vocab('<start>'), *[self.vocab(token) for token in tokens], self.vocab('<end>')]
        if len(caption) == 2:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        caption = torch.tensor(caption)

        return image1, image2, label, caption, caseid


def collate_fn(data):
    # data.sort(key=lambda x: x[-1], reverse=True)
    images1, images2, labels, captions, caseids = zip(*data)

    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    labels = torch.stack(labels, 0)

    max_len = max([len(cap) for cap in captions])
    targets = torch.zeros((len(captions), max_len), dtype=torch.long)
    masks = torch.zeros((len(captions), max_len), dtype=torch.uint8)
    for icap, cap in enumerate(captions):
        l = len(cap)
        targets[icap, :l] = cap
        masks[icap, :l].fill_(1)

    return images1, images2, labels, targets, masks, caseids


class Biview_MultiSent(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, folds, report_fname, vocab_fname, label_fname, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path))
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path))
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        report = self.reports[caseid]
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']
        sents = text.lower().split('.')
        sents = [sent for sent in sents if len(sent.strip()) > 1]
        caption = []
        for isent, sent in enumerate(sents):
            tokens = sent.translate(str.maketrans('', '', string.punctuation)).strip().split()
            if isent==0:
                caption.append([self.vocab('<start>'),*[self.vocab(token) for token in tokens], self.vocab('.')])
            else:
                caption.append([self.vocab('.'),*[self.vocab(token) for token in tokens], self.vocab('.')])
        if caption == []:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        caption[-1].append(self.vocab('<end>'))

        return image1, image2, label, caption, caseid


def sent_collate_fn(data):
    # data.sort(key=lambda x: x[-1], reverse=True)
    images1, images2, labels, captions, caseids = zip(*data)

    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    labels = torch.stack(labels, 0)

    num_sents = [len(cap) for cap in captions]
    sent_lens = [len(sent) for cap in captions for sent in cap]
    max_num_sents = max(num_sents) if len(num_sents) > 0 else 1
    max_sent_lens = max(sent_lens) if len(sent_lens) > 0 else 1
    targets = torch.zeros((len(captions), max_num_sents, max_sent_lens), dtype=torch.long)
    loss_masks = torch.zeros((len(captions), max_num_sents, max_sent_lens))
    update_masks = torch.zeros((len(captions), max_num_sents, max_sent_lens))
    for icap, cap in enumerate(captions):
        for isent, sent in enumerate(cap):
            l = len(sent)
            assert (l > 0)
            targets[icap, isent, :l] = torch.tensor(sent, dtype=torch.long)
            loss_masks[icap, isent, 1:l].fill_(1)
            update_masks[icap, isent, :l-1].fill_(1)

    return images1, images2, labels, targets, loss_masks, update_masks, caseids

class Openi_Sentence_Retrivel(Dataset):
    def __init__(self,
                 vocab,
                 folds,
                 dataset_dir,
                 report_fname,
                 max_sent_lens):
        self.vocab = vocab
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        all_report = []
        for r_id, rp in self.reports.items():
            report = []
            if rp['impression'] is not None:
                report += sent_tokenize(rp['impression'])
            if rp['findings'] is not None:
                report += sent_tokenize(rp['findings'])

            all_report.append(report)

        captions = []
        for irp, rp in enumerate(all_report):
            sents = [sent.lower() for sent in rp]
            for isent, sent in enumerate(sents):
                text = sent.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
                text = text.replace('.', ' .')
                tokens = text.strip().split()
                if len(tokens)>=3:
                    captions.append([self.vocab(token) for token in tokens])

        self.input = torch.zeros(len(captions), max_sent_lens, dtype=torch.long)
        self.masks = torch.zeros(len(captions), max_sent_lens)

        for isent, sent in enumerate(captions):
            l = min(len(sent),max_sent_lens)

            self.input[isent, :l] = torch.tensor(sent[:l], dtype=torch.long)
            self.masks[isent, :l].fill_(1)

        print('Number of sample {}'.format(self.__len__()))
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.masks[index]

class Openi_OneSentence_Retrivel(Dataset):
    def __init__(self,
                 vocab,
                 folds,
                 dataset_dir,
                 report_fname,
                 max_sent_lens):
        self.vocab = vocab
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)

        all_report = []
        for r_id, rp in self.reports.items():
            text = ''
            if rp['impression'] is not None:
                text += rp['impression']
            text += ' '
            if rp['findings'] is not None:
                text += rp['findings']
            all_report.append(text)

        captions = []
        for irp, rp in enumerate(all_report):

            text = rp.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
            text = text.replace('.', ' .')
            tokens = text.strip().split()
            if len(tokens)>=3:
                captions.append([self.vocab(token) for token in tokens])


        # max_sent_lens = max([len(sent) for cap in captions for sent in cap])
        self.input = torch.zeros(len(captions), max_sent_lens, dtype=torch.long)
        self.masks = torch.zeros(len(captions), max_sent_lens)

        for irp, rp in enumerate(captions):
            l = min(len(rp),max_sent_lens)

            self.input[irp, :l] = torch.tensor(rp[:l], dtype=torch.long)
            self.masks[irp, :l].fill_(1)


        print('Number of sample {}'.format(self.__len__()))
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.masks[index]



class Biview_OneSent_retrival(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, folds, report_fname, vocab_fname, label_fname, max_sent_lens, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        self.max_sent_lens = max_sent_lens
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path))
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path))
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        p = random.random()
        if p > 0.5:
            # return the positive pair
            match = torch.tensor([0], dtype=torch.float)
        else:
            # return the negative pair
            new_index = idx
            while new_index == idx:
                new_index = random.randint(0, self.__len__() - 1)
            caseid, _, _ = self.case_list[new_index].split()
            match = torch.tensor([1], dtype=torch.float)
        report = self.reports[caseid]
        report_tensor, mask = self.prepross_report(report)

        return image1, image2, label, report_tensor, mask, match, caseid
    def prepross_report(self, report):
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']

        text = text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
        text = text.replace('.', ' .')
        tokens = text.strip().split()
        if len(tokens) >= 3:
            caption= [self.vocab(token) for token in tokens]
        report_tensor = torch.zeros(self.max_sent_lens, dtype=torch.long)
        mask = torch.zeros(self.max_sent_lens)

        l = min(len(caption), self.max_sent_lens)

        report_tensor[:l] = torch.tensor(caption[:l], dtype=torch.long)
        mask[:l].fill_(1)
        return report_tensor, mask


class Openi_OneSentence_Retrivel_bert(Dataset):
    def __init__(self,
                 tokenizer,
                 folds,
                 dataset_dir,
                 report_fname,
                 max_sent_lens):
        self.tokenizer = tokenizer
        self.case_list = []
        self.max_sent_lens = max_sent_lens
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)

        self.report_text = []
        for r_id, rp in self.reports.items():
            text = ''
            if rp['impression'] is not None:
                text += rp['impression']
            text += ' '
            if rp['findings'] is not None:
                text += rp['findings']
            if len(text.split())>3:
                self.report_text.append(text)
        report_bert = self.tokenizer(self.report_text, padding=True, truncation=True,
                                     max_length=self.max_sent_lens, verbose=False)
        self.input = torch.tensor(report_bert['input_ids'])
        self.masks = torch.tensor(report_bert['attention_mask'])

        print('Number of retrieved report {}'.format(self.__len__()))
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):

        return self.input[index], self.masks[index]

class Openi_Sentence_Retrivel_bert(Dataset):
    def __init__(self,
                 tokenizer,
                 folds,
                 dataset_dir,
                 report_fname,
                 max_sent_lens):
        self.tokenizer = tokenizer
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}_paper.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)

        all_report = []
        for r_id, rp in self.reports.items():
            text = ''
            if rp['impression'] is not None:
                text += rp['impression']
            text += ' '
            if rp['findings'] is not None:
                text += rp['findings']
            sents = text.lower().split('.')
            if len(sents) > 2:
                all_report += sents

        report_bert = self.tokenizer(all_report, padding=True, truncation=True,
                                     max_length=max_sent_lens, verbose=False)
        self.input = torch.tensor(report_bert['input_ids'])
        self.masks = torch.tensor(report_bert['attention_mask'])

        print('Number of retrieved sentence {}'.format(self.__len__()))
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.masks[index]

class Biview_OneSent_retrival_bert(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, txt, report_fname, tokenizer, label_fname, max_sent_lens, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        self.tokenizer = tokenizer
        self.max_sent_lens = max_sent_lens

        with open(os.path.join(dataset_dir, txt)) as f:
            self.case_list += f.read().splitlines()

        with open(report_fname) as f:
            self.reports = json.load(f)

        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

        self.report_text = []
        self.r_id = {}
        for idx, (r_id, rp) in enumerate(self.reports.items()):
            text = ''
            if rp['impression'] is not None:
                text += rp['impression']
            text += ' '
            if rp['findings'] is not None:
                text += rp['findings']
            self.report_text.append(text)
            self.r_id[r_id]=idx
        report_bert = self.tokenizer(self.report_text, padding=True, truncation=True,
                                     max_length=self.max_sent_lens, verbose=False)
        self.input = torch.tensor(report_bert['input_ids'])
        self.masks = torch.tensor(report_bert['attention_mask'])
    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path))
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path))
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        p = random.random()
        if p > 0.5:
            # return the positive pair
            match = torch.tensor([0], dtype=torch.float)
        else:
            # return the negative pair
            new_index = idx
            while new_index == idx:
                new_index = random.randint(0, self.__len__() - 1)
            caseid, _, _ = self.case_list[new_index].split()
            match = torch.tensor([1], dtype=torch.float)
        report_tensor, mask = self.input[self.r_id[caseid]], self.masks[self.r_id[caseid]]

        return image1, image2, label, report_tensor, mask, match, caseid

class MIMIC_OneSentence_Retrivel_bert(Dataset):
    def __init__(self,
                 tokenizer,
                 txt,
                 dataset_dir,
                 report_fname,
                 max_sent_lens):
        self.tokenizer = tokenizer
        self.case_list = []
        self.max_sent_lens = max_sent_lens
        with open(txt) as f:
            self.case_list += f.read().splitlines()

        with open(report_fname) as f:
            self.reports = json.load(f)

        self.report_text = []
        for r_id, rp in tqdm(self.reports.items()):
            text = ''
            if rp['impression'] is not None:
                text += rp['impression']
            text += ' '
            if rp['findings'] is not None:
                text += rp['findings']
            if len(text.split())>3:
                self.report_text.append(text)
        report_bert = self.tokenizer(self.report_text, padding=True, truncation=True,
                                     max_length=self.max_sent_lens, verbose=False)
        self.input = torch.tensor(report_bert['input_ids'])
        self.masks = torch.tensor(report_bert['attention_mask'])

        print('Number of retrieved report {}'.format(self.__len__()))
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):

        return self.input[index], self.masks[index]

class MIMIC_Biview_OneSent_retrival_bert(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, txt, report_fname, tokenizer, label_fname, max_sent_lens, train=True):

        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        self.tokenizer = tokenizer
        self.max_sent_lens = max_sent_lens
        print('Creating MIMIC {} dataset'.format(self.phase))
        with open(txt) as f:
            self.case_list += f.read().splitlines()

        with open(report_fname) as f:
            self.reports = json.load(f)

        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

        self.report_text = []
        self.r_id = {}
        for idx, (r_id, rp) in enumerate(tqdm(self.reports.items())):
            text = ''
            if rp['impression'] is not None:
                text += rp['impression']
            text += ' '
            if rp['findings'] is not None:
                text += rp['findings']
            text = text.replace('\n', '')
            self.report_text.append(text)
            self.r_id[r_id]=idx
        report_bert = self.tokenizer(self.report_text, padding=True, truncation=True,
                                     max_length=self.max_sent_lens, verbose=False)

        self.input = report_bert['input_ids']
        self.masks = report_bert['attention_mask']
        print('Creating {} dataset: {}'.format(self.phase, self.__len__()))
    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path)).convert('RGB')
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path)).convert('RGB')
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        p = random.random()
        if p > 0.5:
            # return the positive pair
            match = torch.tensor([0], dtype=torch.float)
        else:
            # return the negative pair
            new_index = idx
            while new_index == idx:
                new_index = random.randint(0, self.__len__() - 1)
            caseid, _, _ = self.case_list[new_index].split()
            match = torch.tensor([1], dtype=torch.float)
        report_tensor, mask = torch.tensor(self.input[self.r_id[caseid]]), \
                              torch.tensor(self.masks[self.r_id[caseid]])

        return image1, image2, label, report_tensor, mask, match, caseid


class MIMIC_Biview_MultiSent(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, txt, report_fname, vocab_fname, label_fname, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        with open(txt) as f:
            self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                Equalize(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path)).convert('RGB')
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path)).convert('RGB')
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        report = self.reports[caseid]
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']
        sents = text.lower().split('.')
        sents = [sent for sent in sents if len(sent.strip()) > 1]
        caption = []
        for isent, sent in enumerate(sents):
            tokens = sent.translate(str.maketrans('', '', string.punctuation)).strip().split()
            if isent==0:
                caption.append([self.vocab('<start>'),*[self.vocab(token) for token in tokens], self.vocab('.')])
            else:
                caption.append([self.vocab('.'),*[self.vocab(token) for token in tokens], self.vocab('.')])
        if caption == []:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        caption[-1].append(self.vocab('<end>'))

        return image1, image2, label, caption, caseid

class MIMIC_Biview_OneSent(Dataset):

    def __init__(self, phase, dataset_dir, image_dir, txt, report_fname, vocab_fname, label_fname, train=True):
        self.phase = phase
        self.image_dir = image_dir
        self.case_list = []
        with open(txt) as f:
            self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((512, 512), scale=(0.8,1.2)),
                transforms.RandomHorizontalFlip(),
                Equalize(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                Equalize(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(os.path.join(self.image_dir,img1_path)).convert('RGB')
        image1 = self.transform(image1)
        image2 = Image.open(os.path.join(self.image_dir,img2_path)).convert('RGB')
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        report = self.reports[caseid]
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']
        text = text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
        text = text.replace('.', ' .')
        tokens = text.strip().split()
        caption = [self.vocab('<start>'), *[self.vocab(token) for token in tokens], self.vocab('<end>')]
        if len(caption) == 2:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        caption = torch.tensor(caption)

        return image1, image2, label, caption, caseid
if __name__ == '__main__':
    pass
from PIL import Image, ImageOps
from torchvision import transforms
def decode(vocab, seq, caseids):
    decoded_dict = {}
    for idx, caseid in enumerate(caseids):
        decoded_dict[caseid] = ['']
        pred = seq[idx].detach().cpu()
        end_report = False
        for isent in range(pred.size(0)):
            words = []

            if end_report:
                break

            for wid in pred[isent].tolist():
                w = vocab.idx2word[wid]
                if w == '<start>' or w == '<pad>':
                    continue

                if w == '<end>':
                    end_report = True
                    break

                words.append(w)

            decoded_dict[caseid][0] += ' '.join(words)
            decoded_dict[caseid][0] += ' '
    return decoded_dict

def decode_sent(vocab, seq, caseids):
    decoded_dict = {}
    for idx, caseid in enumerate(caseids):
        decoded_dict[caseid] = ['']
        pred = seq[idx].detach().cpu()
        words = []
        for wid in pred.tolist():
            w = vocab.idx2word[wid]
            if w == '<pad>':
                continue
            if w == '<end>':
                break
            words.append(w)
        decoded_dict[caseid][0] += ' '.join(words)
        decoded_dict[caseid][0] += ' '
    return decoded_dict

def decode_sent_bert(tokenizer, seq, caseids):
    decoded_dict = {}
    for idx, caseid in enumerate(caseids):
        decoded_dict[caseid] = ['']
        pred = seq[idx].detach().cpu()
        decoded_dict[caseid][0] += tokenizer.decode(pred, skip_special_tokens=True)

    return decoded_dict

def decode_const( caseids):
    const = 'no acute cardiopulmonary abnormality . the heart size and pulmonary vascularity are within normal limits . there is no focal consolidation pleural effusion or pneumothorax . no focal consolidation pleural effusion or pneumothorax . no pneumothorax or pleural effusion .'
    decoded_dict = {}
    for idx, caseid in enumerate(caseids):
        decoded_dict[caseid] = [const]
    return decoded_dict
def print_example(gt, pre, num=5):
    keys = list(gt.keys())
    for key in keys[:num]:
        print("GT:{}\nPred:{}\n".format(gt[key][0],pre[key][0]))

class Equalize(object):
    def __init__(self):
        pass

    def __call__(self, image):
        equ = ImageOps.equalize(image, mask=None)
        return equ
if __name__ == '__main__':
    img = Image.open('F:\openi\\NLMCXR_png\CXR1_1_IM-0001-4001.png')
    # img.show()
    eq = transforms.RandomResizedCrop((512,512), scale=(0.8,1.2))(img)
    eq.show()

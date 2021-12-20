import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np

from model import create_model
from options import Options


class BaseSolver(object):
    """docstring for BaseSolver"""

    def __init__(self):
        super(BaseSolver, self).__init__()

    def initialize(self, opt):
        self.opt = opt

        self.CK_FACIAL_EXPRESSION = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.OC_FACIAL_EXPRESSION = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
        self.JA_FACIAL_EXPRESSION = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    def init_test_setting(self, opt):
        # hard code some params
        opt.visdom_display_id = 0
        opt.serial_batches = True

        model = create_model(opt)
        model.set_eval()
        return model


class ResFaceClsSolver(BaseSolver):
    """docstring for ResFaceClsSolver"""

    def __init__(self):
        super(ResFaceClsSolver, self).__init__()

    def test_networks(self, opt, batch):
        # go through all the dataset and generate map
        model = self.init_test_setting(opt)
        pred_cls_list = []
        with torch.no_grad():
            model.feed_batch(batch)
            model.forward()

            pred_cls = model.pred_cls.detach().cpu().numpy()
            pred_cls = np.argmax(pred_cls, axis=1)  # predicted class
            pred_cls_list.extend(pred_cls)

        # print("Predict label list: ", pred_cls_list)
        # for i in range(len(pred_cls_list)):
        #     print("[", i + 1, "]", pred_cls_list[i], self.CK_FACIAL_EXPRESSION[pred_cls_list[i]])
        #
        # print("**********")
        return pred_cls


img2tensor = transforms.Compose(
    [  # Để thực hiện nhiều phép biến đổi trên dữ liệu đầu vào, transforms hỗ trợ hàm compose để gộp các transforms lại.
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Tham số đầu là mean, tham số sau là std => T.Normalize(mean, std)
    ])


def transform_image(image):
    img = transforms.functional.resize(image, 320)
    img = transforms.functional.five_crop(img, 299)[-1]  # final_size=299
    return img2tensor(img)


if __name__ == '__main__':
    options = Options().parse()
    input_img = Image.open('datasets/AffectNet/imgs/train_set/0.jpg').convert('RGB')
    input_img_2 = Image.open('datasets/AffectNet/imgs/train_set/0.jpg').convert('L')
    instance = ResFaceClsSolver()
    instance.initialize(options)

    t_img = transform_image(input_img).unsqueeze(1)
    t_img_2 = transform_image(input_img_2).unsqueeze(1)

    batch = {
        'img_tensor': t_img,
        'img_tensor_gray': t_img_2,
        'img_res_tensor': t_img_2,
        'img_path': 'datasets'
    }

    rs = instance.test_networks(options, batch)
    print(rs)

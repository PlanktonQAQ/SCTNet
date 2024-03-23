import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import glob
from torchvision import transforms as T
import numpy as np
from PIL import Image
import cv2
from utils.augmentations import FastBaseTransform
from collections import OrderedDict
import cv2
from options.test_options import TestOptions
import torchvision.utils as vutils
from PIL import Image
from util import util
from data.base_dataset import get_transform_lab

# seg model
from data import cfg
from yolact import Yolact
from u2net import U2NETP # small version u2net 4.7 MB
from layers.output_utils import postprocess

# color transfer model
import torch.nn.functional as F



opt = TestOptions().parse()
opt.checkpoints_dir = 'colorTransfer/checkpoint/'
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True



def get_lightness(src):
    # calculate the lightness of the image
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = np.array(hsv_image[:, :, 2].mean()).astype(np.float32)

    return lightness

def imgBrightness(fimg, bimg, cimg, s_mask):
    """
    1. get image's lightness value to build the different between fore-img and back img
    2. compute the c value between the image's lightness is the adaptive operation
    """
    a = 10
    b = 180
    s_mask = np.array(s_mask.squeeze(0).cpu()).astype(np.uint8)
    fimg = fimg*s_mask
    bimg = bimg*(1-s_mask)
    light_f = get_lightness(fimg)
    light_b = get_lightness(bimg)
    light_c = get_lightness(cimg)
    w1, w2 = 1, 1

    # Version 2
    if light_f < a:
        w1 = (light_f / (light_b + light_f) * 1.0) + 1  # too dark
    elif light_f > b:
        w1 = (light_f / (light_b + light_f) * 1.0)  # too light
    else:
        w1 = 1
    if light_b < a:
        w2 = (light_b / (light_b + light_f) * 1.0) + 1  # too dark
    elif light_b > b:
        w2 = (light_b / (light_b + light_f) * 1.0)  # too light
    else:
        w2 = 1
    print('foreground_image_brightness:', light_f, 'background_image_brightness:', light_b, 'combine_image_brightness:',light_c, 'image light weight:', w1, w2)
    rst = cv2.add(w1*fimg, w2*bimg, dtype=cv2.CV_32F)
    return rst


def save_combine_image(visuals, input_file, target_file, save_dir, s_mask):
    global out_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # test convert
    grid_img_forge = vutils.make_grid(visuals['forge_ground_img'], nrow=1, padding=0)
    grid_img_back = vutils.make_grid(visuals['back_ground_img'], nrow=1, padding=0)
    pil_img_forge = T.functional.to_pil_image(grid_img_forge)
    pil_img_back = T.functional.to_pil_image(grid_img_back)
    cv2_img_forge = cv2.cvtColor(np.array(pil_img_forge), cv2.COLOR_RGB2BGR)
    cv2_img_back = cv2.cvtColor(np.array(pil_img_back), cv2.COLOR_RGB2BGR)
    # cv2.imwrite('results_test/forge.png', cv2_img_forge)
    # cv2.imwrite('results_test/back.png', cv2_img_back)

    try:
        id_input = str(input_file).split('.')[0].split('/')[-1]
        id_t = str(target_file).split('.')[0].split('/')[-1]
        id = id_input+'_' + id_t
        name = str(id) + '_iat'

        # make sure the size of the image is the same
        cimg = cv2.add(cv2_img_forge, cv2_img_back)
        # save_dir_cmb = summary_dir + 'Combine/'
        # # Lightness Adaptive Folder
        # save_dir_IAT = summary_dir + 'IAT/'
        # if not os.path.exists(save_dir_IAT):
        #     os.makedirs(save_dir_IAT)

        com_img = imgBrightness(cv2_img_forge, cv2_img_back, cimg, s_mask)
        out_path = '{}/{}.png'.format(save_dir, name)
        cv2.imwrite(out_path, com_img)
    except IOError as e:
        print('combine the predground and background get problem')
    return out_path



def data_load(source_path, target_path):

    source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)

    # data augmentation
    # if source_img.shape[0] > source_img.shape[1]:
    #     source_img = cv2.transpose(source_img)
    # if target_img.shape[0] > target_img.shape[1]:
    #     target_img = cv2.transpose(target_img)
    # source_img = cv2.resize(source_img, (600, 400))    # 512, 384 / 768, 576/ 1024, 768
    # target_img = cv2.resize(target_img, (600, 400))
    # max 700 px with aspect ratio
    if (source_img.shape[0]>700) or (source_img.shape[1]>700):
        aspect_ratio = source_img.shape[0] / source_img.shape[1]
        if source_img.shape[0] > source_img.shape[1]:
            source_img = cv2.resize(source_img, (int(700/aspect_ratio),700))
        else:
            source_img = cv2.resize(source_img, (700,int(700*aspect_ratio)))
    # max 700 px with aspect ratio
    if (target_img.shape[0]>700) or (target_img.shape[1]>700):
        aspect_ratio = target_img.shape[0] / target_img.shape[1]
        if target_img.shape[0] > target_img.shape[1]:
            target_img = cv2.resize(target_img, (int(700/aspect_ratio),700))
        else:
            target_img = cv2.resize(target_img, (700,int(700*aspect_ratio)))


    source_batch = torch.from_numpy(source_img).float()
    target_batch = torch.from_numpy(target_img).float()
    source_batch = FastBaseTransform()(source_batch.unsqueeze(0))
    target_batch = FastBaseTransform()(target_batch.unsqueeze(0))

    return source_batch, target_batch, source_img, target_img, source_path, target_path

def get_mask(seg_result, size, d, source_img, target_img):
    masks_out = []
    masks_ins_out = []
    for i in range(len(seg_result)):
        h = size[i][1]
        w = size[i][2]
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        temp = []
        temp.append(seg_result[i])
        t = postprocess(temp, w, h, visualize_lincomb=False,crop_masks=True,score_threshold=0)
        cfg.rescore_bbox = save
        mask_sal_resize = F.interpolate(d[i].unsqueeze(1), size=(h, w), mode='bilinear').squeeze(1)
        mask_sal_resize = mask_sal_resize > 0.08
        if len(t[3]) < 1:
            mask_ins = mask_sal_resize
        elif len(t[3]) < 15:
            print(len(t[3]))
            mask_ins = t[3]
            mask_ins = mask_ins > 0.5
        else:
            mask_ins = t[3][:15, :, :]
            mask_ins = mask_ins > 0.5
        # calculate the iou
        intersection = mask_sal_resize & mask_ins
        union = mask_sal_resize | mask_ins
        intersection_area = intersection.sum(dim=(1, 2))
        union_area = union.sum(dim=(1, 2))
        iou = intersection_area / union_area
        index_sal = iou.argmax()
        masks_ins_out.append(mask_ins[0])
        masks_out.append(mask_ins[index_sal])
    return masks_out


if __name__ == '__main__':

    # define the data to eval
    content_dir = 'dataset/Source/'
    style_dir = 'dataset/Reference/'
    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    content_paths = sorted(glob.glob(content_dir + '*'))
    style_paths = sorted(glob.glob(style_dir + '*'))
    
    # load the color transfer model
    transform_type = get_transform_lab(opt)
    model_path = 'checkpoint/FNet_2_color_checkpoint.tar'
    ColNet = torch.load(model_path)
    ColNet.IRN.eval()

    # load the segment model
    Seg_weight = 'checkpoint/yolact_base_54_800000.pth'
    SegNet = Yolact()
    SegNet.load_weights(Seg_weight)
    SegNet.cuda()
    SegNet.eval()

    # Salience init
    print("...load U2NEP---4.7 MB")
    u2net = U2NETP(3,1)
    model_dir = 'checkpoint/u2netp.pth'
    sal_transform = T.Compose([
                T.Resize((320,320), interpolation=T.InterpolationMode.BICUBIC),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(model_dir))
        # net = nn.DataParallel(net)
        u2net.cuda()
        # print("Gpu")
    else:
        u2net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        # print("Cpu")
    u2net.eval()

    for i in range(len(content_paths)):
        content_path = content_paths[i]
        style_path = style_paths[i]

        source_batch, target_batch, source_img, target_img, source_path, target_path = data_load(content_path, style_path)
        print(source_path)
        print(target_path)

        source_batch = source_batch.squeeze(1).cuda()
        target_batch = target_batch.squeeze(1).cuda()
        source_img = torch.from_numpy(source_img).float().cuda().unsqueeze(0)
        target_img = torch.from_numpy(target_img).float().cuda().unsqueeze(0)

        with torch.no_grad():
            # Segment
            img_size = []
            img_size.append(source_img.shape)
            img_size.append(target_img.shape)
            d,_,_,_,_,_,_ = u2net(torch.cat((source_batch, target_batch), 0))
            d = util.normPRED(d[:,0,:,:]).unsqueeze(1)
            masks = SegNet(torch.cat((source_batch, target_batch), 0))
            masks = get_mask(masks, img_size, d, source_img, target_img)
            source_ins = source_img.squeeze() * masks[0].unsqueeze(0).permute(1,2,0)
            source_back = source_img.squeeze() - source_ins
            target_ins = target_img.squeeze() * masks[1].unsqueeze(0).permute(1,2,0)
            target_back = target_img.squeeze() - target_ins

            # Color Transfer
            source_ins_col = np.array(source_ins.detach().cpu()).astype('uint8')
            source_ins_col = Image.fromarray(source_ins_col[:, :, ::-1])
            target_ins_col = np.array(target_ins.detach().cpu()).astype('uint8')
            target_ins_col = Image.fromarray(target_ins_col[:, :, ::-1])
            source_back_col = np.array(source_back.detach().cpu()).astype('uint8')
            source_back_col = Image.fromarray(source_back_col[:, :, ::-1])
            target_back_col = np.array(target_back.detach().cpu()).astype('uint8')
            target_back_col = Image.fromarray(target_back_col[:, :, ::-1])
            source_ins_col = transform_type(source_ins_col).float()
            target_ins_col = transform_type(target_ins_col).float()
            source_ins_col = source_ins_col.unsqueeze(0).cuda()
            target_ins_col = target_ins_col.unsqueeze(0).cuda()
            source_back_col = transform_type(source_back_col).float()
            target_back_col = transform_type(target_back_col).float()
            source_back_col = source_back_col.unsqueeze(0).cuda()
            target_back_col = target_back_col.unsqueeze(0).cuda()
            my_dict = OrderedDict()
            ins_out = ColNet.forward(source_ins_col, target_ins_col)
            my_dict['forge_ground_img'] = ins_out['res_img']
            back_out = ColNet.forward(source_back_col, target_back_col)
            my_dict['back_ground_img'] = back_out['res_img']
            com_image_path = save_combine_image(
                my_dict, source_path, target_path, save_dir, masks[0].unsqueeze(0).permute(1,2,0))






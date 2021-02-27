import cv2
import numpy as np
import os
import torch

###   PATHS   ###
path1 = "./benchmark/paintsChainer/line/0.png"
path_ref = "./benchmark/paintsChainer/ref/0.png"
image_out_path = "./benchmark/paintsChainer/out/0_0.jpg"
path_model = "./benchmark/paintsChainer/model.ts"

def preprocess(path1, path_ref, s_size=128):
    def cvt2YUV(img):
        (major, minor, _) = cv2.__version__.split(".")
        if major == '3':
            img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
        else:
            img = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
        return img
    s_size = 128

    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    print("load:" + path1, os.path.isfile(path1), image1 is None)
    image1 = np.asarray(image1, np.float32)

    _image1 = image1.copy()
    if image1.shape[0] < image1.shape[1]:
        s0 = s_size
        s1 = int(image1.shape[1] * (s_size / image1.shape[0]))
        s1 = s1 - s1 % 16
        _s0 = 4 * s0
        _s1 = int(image1.shape[1] * ( _s0 / image1.shape[0]))
        _s1 = (_s1+8) - (_s1+8) % 16
    else:
        s1 = s_size
        s0 = int(image1.shape[0] * (s_size / image1.shape[1]))
        s0 = s0 - s0 % 16
        _s1 = 4 * s1
        _s0 = int(image1.shape[0] * ( _s1 / image1.shape[1]))
        _s0 = (_s0+8) - (_s0+8) % 16

    _image1 = image1.copy()
    _image1 = cv2.resize(_image1, (_s1, _s0),
                            interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, (s1, s0), interpolation=cv2.INTER_AREA)

    # image is grayscale
    if image1.ndim == 2:
        image1 = image1[:, :, np.newaxis]
    if _image1.ndim == 2:
        _image1 = _image1[:, :, np.newaxis]

    image1 = np.insert(image1, 1, -512, axis=2)
    image1 = np.insert(image1, 2, 128, axis=2)
    image1 = np.insert(image1, 3, 128, axis=2)

    # add color ref image

    image_ref = cv2.imread(path_ref, cv2.IMREAD_UNCHANGED)
    image_ref = cv2.resize(image_ref, (image1.shape[1], image1.shape[
                            0]), interpolation=cv2.INTER_NEAREST)
    try:
        b, g, r, a = cv2.split(image_ref)
    except:
        b, g, r = cv2.split(image_ref)
        a = np.zeros_like(r)
        
    image_ref = cvt2YUV( cv2.merge((b, g, r)) )

    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            if a[x][y] != 0:
                for ch in range(3):
                    image1[x][y][ch + 1] = image_ref[x][y][ch]

    return (
        torch.from_numpy(image1.transpose(2, 0, 1)).unsqueeze(0),
        torch.from_numpy(_image1.transpose(2, 0, 1)).unsqueeze(0)
    )

def save_as_img(array, name):
    array = array.transpose(1, 2, 0)
    array = array.clip(0, 255).astype(np.uint8)
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor(array, cv2.COLOR_YUV2RGB)
    else:
        img = cv2.cvtColor(array, cv2.COLOR_YUV2BGR)
    cv2.imwrite(name, img)

with torch.no_grad():
    model = torch.jit.load(path_model)

    sample = preprocess(path1, path_ref)
    img = model(sample[0].cuda(), sample[1].cuda())

    save_as_img(img[0].cpu().numpy(), image_out_path)
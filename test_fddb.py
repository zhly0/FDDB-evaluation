from __future__ import print_function
import os
import argparse
import numpy as np
import cv2
from centerface_v3 import CenterFace

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('--dataset', default=r'F:\face_detection\centerface-master\centerface-master\FDDB', type=str, help='dataset')
parser.add_argument('--confidence_threshold', default=0.1, type=float, help='confidence_threshold')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


if __name__ == '__main__':
    # save file
    fw = open(os.path.join(args.dataset + '/fddb_dets0.txt'), 'w')

    # testing dataset
    testset_folder = os.path.join(args.dataset, 'originalPics/')
    testset_list = os.path.join(args.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    resize = 1

    #_t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name + '.jpg'
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        centerface = CenterFace()
        dets = centerface(img, 0.1,0.3)

        # save dets
        print(i)
        if 1:
            fw.write('{:s}\n'.format(img_name))
            fw.write('{:.1f}\n'.format(dets.shape[0]))
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                # fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
                fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))
        #print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # save image
            if not os.path.exists("./results1/"):
                os.makedirs("./results1/")
            name = "./results1/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

    fw.close()

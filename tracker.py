# Created by if-pc at 2018/12/11
# author: xuyouze

import numpy as np
import cv2
import os


class Tracker(object):
    def __init__(self, folder, ground_truth_file, delimiter, is_color, video_name, track_window="tracker",
                 num_particle=100, weight=0.009):

        self.file_folder = folder
        self.ground_truth = np.loadtxt(ground_truth_file, dtype=int, delimiter=delimiter)
        self.track_window = track_window  # img windows name
        self.num_particle = num_particle  # the number of particle
        self.threshold = weight  # weight for resample the particle
        self.iscolor = is_color  # whether the image is color image
        self.target_feature = None
        self.target_center = None
        self.target_region = None
        self.target_roi = None  # target image
        self.particles = None  # the list of particle  for every particle is [left_x,left_y, right_x,right_y]
        self.pic_w = None  # the width of image
        self.pic_h = None  # the height of image
        self.output_name = video_name  # output video name

    def run(self):
        cv2.namedWindow(self.track_window)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        # for loop all image
        for i, name in enumerate(sorted(os.listdir(self.file_folder))):
            file_name = os.path.join(self.file_folder, name)
            img = cv2.imread(file_name)

            # init all the params
            if not i:
                # get ground_truth box
                self.get_roi_feature(img)
                self.pic_h, self.pic_w = img.shape[:2]
                # init particles
                self.particles = self.init_particle(self.target_center[0], self.target_center[1], self.num_particle)
                out = cv2.VideoWriter(self.output_name + '.mp4', fourcc, 15.0, (self.pic_w, self.pic_h), True)
                continue

            # calculate the similarity
            similarity = self.likelihood(img)

            # re-sample，get the best and mean box
            mean_box, best_box = self.re_sample(similarity)

            left_x, left_y, right_x, right_y = best_box
            mean_left_x, mean_left_y, mean_right_x, mean_right_y = mean_box
            x_t, y_t, w_t, h_t = self.ground_truth[i]

            font = cv2.FONT_HERSHEY_SIMPLEX
            # mean
            img_result = cv2.rectangle(img, (mean_left_x, mean_left_y), (mean_right_x, mean_right_y), color=(255, 0, 0),
                                       thickness=2)
            cv2.putText(img_result, "mean", (10, 40), font, 0.6, (255, 0, 0), 1)
            # ground truth
            cv2.rectangle(img_result, (x_t, y_t), (x_t + w_t, y_t + h_t), color=(0, 255, 0), thickness=2)
            cv2.putText(img_result, "truth", (10, 60), font, 0.6, (0, 255, 0), 1)
            # best
            cv2.rectangle(img_result, (left_x, left_y), (right_x, right_y), color=(0, 0, 255), thickness=2)
            cv2.putText(img_result, "best", (10, 20), font, 0.6, (0, 0, 255), 1)

            cv2.imshow(self.track_window, img_result)
            out.write(img_result)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                pass

        out.release()
        cv2.destroyAllWindows()

    def get_roi_feature(self, img):
        """
        get target feature
        :param img:
        :return:
        """
        self.target_region = self.ground_truth[0]
        x, y, w, h = self.target_region
        self.target_center = x + 1 / 2 * w, y + 1 / 2 * h
        # get the target image
        self.target_roi = img[int(y):int(y + h), int(x):int(x + w)]

        # calculate hist
        self.target_feature = self.cal_feature(self.target_roi)

    def cal_feature(self, img):
        """
        calculate hist
        :param img:
        :return hist of the image:
        """
        if self.iscolor:
            # hsv hist
            hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 计算颜色直方图
            roi_hist = cv2.calcHist([hsv_roi], [0], None, [20], [0, 180])
        else:
            # grey hist
            roi_hist = cv2.calcHist([img], [0], None, [20], [0, 255])
        # normalize the hist
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # sift feature
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp, des = sift.detectAndCompute(img, None)
        # return roi_hist, des
        return roi_hist

    def softmax(self, x):
        """Compute the softmax in a numerically stable way."""

        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def likelihood(self, img):
        """
        calculate similarity between the target feature and particle feature
        the larger the value of similarity, the less similar the two features are.
        :param img:
        :return the list of similarity:
        """
        target_hist_feature = self.target_feature

        similarity = np.zeros(self.num_particle)
        for i, particle in enumerate(self.particles):
            left_x, left_y, right_x, right_y = particle
            particle_hist = self.cal_feature(img[left_y:right_y, left_x:right_x])
            # use BHATTACHARYYA distance
            similarity[i] = cv2.compareHist(target_hist_feature, particle_hist, method=cv2.HISTCMP_BHATTACHARYYA)
            if np.isnan(similarity[i]):
                similarity[i] = 0.5

        similarity = self.softmax(similarity)
        return similarity

    def init_particle(self, center_x, center_y, num):
        """
        according to the center_x and center_y ,initialize num particles
        :param center_x:
        :param center_y:
        :param num:
        :return:
        """
        x, y, w, h = self.target_region

        par_range = (w + h) / 16.0
        x_centers = par_range * np.random.uniform(-1.0, 1.0, num) + center_x
        y_centers = par_range * np.random.uniform(-1.0, 1.0, num) + center_y

        left_x, left_y, right_x, right_y = x_centers - 0.5 * w, y_centers - 0.5 * h, x_centers + 0.5 * w, y_centers + 0.5 * h

        left_x = np.clip(left_x, 0, self.pic_w - 1).astype(np.int)
        right_x = np.clip(right_x, 0, self.pic_w).astype(np.int)
        left_y = np.clip(left_y, 0, self.pic_h - 1).astype(np.int)
        right_y = np.clip(right_y, 0, self.pic_h).astype(np.int)
        return np.stack((left_x, left_y, right_x, right_y), axis=1)

    def re_sample(self, similarity):
        """
        according to the threshold update the particle, and return the mean and best position of particle.
        :param similarity:
        :return mean_box and best_box:
        """
        indexes = np.where(similarity > self.threshold)[0]
        num = indexes.shape[0]

        min_index = np.argmin(similarity)

        best_box = self.particles[min_index].copy()
        x_center, y_center = np.sum(best_box.reshape(2, 2), axis=0) / 2.0
        self.particles[indexes] = self.init_particle(int(x_center), int(y_center), num)

        mean_box = (np.sum(self.particles, axis=0) / self.num_particle).astype(np.int)

        return mean_box, best_box

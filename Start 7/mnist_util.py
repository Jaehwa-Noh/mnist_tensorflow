import cv2

def split_train_validation(train_image, train_label, percent):
    split_percent = int(len(train_image) * percent / 100)

    tmp_train_image = train_image[:split_percent]
    tmp_train_label = train_label[:split_percent]
    tmp_validation_image = train_image[split_percent:]
    tmp_validation_label = train_label[split_percent:]

    return tmp_train_image, tmp_train_label,\
        tmp_validation_image, tmp_validation_label


def load_real_data(path, number):
    real_data_list = list()
    tmp_img = 11
    for repeat in range(number):


        try:
            tmp_img = cv2.imread(path+'data{}.png'.format(repeat+1), cv2.IMREAD_GRAYSCALE)

        except:
            print('error occur.')
            exit()

        tmp_img_height, tmp_img_width = tmp_img.shape
        if tmp_img_height != 28 and tmp_img_width != 28:
            tmp_img = cv2.resize(tmp_img, (28, 28))
        print(tmp_img.shape)
        pre_treat_img = (tmp_img > 10).astype(int)
        real_data_list.append(pre_treat_img)

    return real_data_list

#

def split_train_validation(train_image, train_label, percent):
    split_percent = int(len(train_image) * percent / 100)

    tmp_train_image = train_image[:split_percent]
    tmp_train_label = train_label[:split_percent]
    tmp_validation_image = train_image[split_percent:]
    tmp_validation_label = train_label[split_percent:]

    return tmp_train_image, tmp_train_label,\
        tmp_validation_image, tmp_validation_label




#

import numpy as np


def mix_up(batch_images, batch_labels, p=0.5):
    batch_size = batch_images.shape[0]
    n_classes = batch_labels.shape[1]
    image_size = batch_images.shape[1]

    images = []
    labels = []
    for j in range(batch_size):
        prob = np.random.uniform(0, 1) <= p
        prob = float(prob)
        k = np.random.randint(0, batch_size)
        beta_dist = np.random.uniform(0, 1)
        weight = beta_dist * prob
        img1 = batch_images[j, :]
        img2 = batch_images[k, :]
        images.append((1 - weight) * img1 + weight * img2)

        lab1 = batch_labels[j, :]
        lab2 = batch_labels[k, :]
        labels.append((1 - weight) * lab1 + weight * lab2)

    output_images = np.reshape(np.stack(images), (batch_size, image_size, image_size, 3))
    output_labels = np.reshape(np.stack(labels), (batch_size, n_classes))
    return output_images, output_labels


def cut_mix(batch_images, batch_labels, p=0.5):
    batch_size = batch_images.shape[0]
    n_classes = batch_labels.shape[1]
    image_size = batch_images.shape[1]

    images = []
    labels = []
    for j in range(batch_size):
        prob = np.random.uniform(0, 1) <= p
        k = j
        while k == j:
            k = np.random.randint(0, batch_size)
        x = np.random.randint(0, image_size)
        y = np.random.randint(0, image_size)
        beta_dist = np.random.uniform(0, 1)
        width = int(image_size * np.sqrt(1 - beta_dist)) * prob
        ya = max(0, y - width // 2)
        yb = min(image_size, y + width // 2)
        xa = max(0, x - width // 2)
        xb = min(image_size, x + width // 2)

        one = batch_images[j, ya:yb, 0:xa, :]
        two = batch_images[k, ya:yb, xa:xb, :]
        three = batch_images[j, ya:yb, xb:image_size, :]
        middle = np.concatenate([one, two, three], axis=1)
        img = np.concatenate([batch_images[j, 0:ya, :, :], middle, batch_images[j, yb:image_size, :, :]], axis=0)
        images.append(img)

        weight = width * width / image_size / image_size
        lab1 = batch_labels[j]
        lab2 = batch_labels[k]
        labels.append((1 - weight) * lab1 + weight * lab2)

    output_images = np.reshape(np.stack(images), (batch_size, image_size, image_size, 3))
    output_labels = np.reshape(np.stack(labels), (batch_size, n_classes))
    return output_images, output_labels


def cut_out(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    img_h, img_w, img_c = image.shape
    p_1 = np.random.rand()
    if p_1 > p:
        return image

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    image[top:top + h, left:left + w, :] = c
    return image

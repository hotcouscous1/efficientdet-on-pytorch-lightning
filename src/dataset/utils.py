from src.__init__ import *


def make_one_hot(
        num_classes: int,
        category_id: int
) -> Tensor:

    return torch.from_numpy(np.eye(num_classes, dtype='int8')[category_id]).to(device)



def category_filter(
        original_cat: Union[list, tuple],
        missing_cat: Union[list, tuple]
) -> Dict:

    valid_cat = list(filter(lambda c: c not in missing_cat, original_cat))
    cat_table = {c: i for i, c in enumerate(valid_cat)}
    return cat_table



def fill_empty_category(
        category_id: int,
        missing_cat: Union[list, tuple],
        start_id: int
) -> int:

    category_id += start_id

    for missing_id in missing_cat:
        if category_id >= missing_id:
            category_id += 1
        else:
            break

    return category_id



def imagenet_fill():
    return tuple([round(255 * m) for m in (0.485, 0.456, 0.406)])



def make_mini_batch(
        sample
) -> Tuple[Tensor, Tensor]:

    images, labels, target_nums = [], [], []

    zero_labels = []
    max_target_num = 0

    for image, label in sample:
        images.append(image)
        labels.append(label)
        target_num = label.size(0)
        target_nums.append(target_num)
        max_target_num = max(max_target_num, target_num)

    for label in labels:
        target_num = label.size(0)
        zero_fill = torch.zeros((max_target_num - target_num, label.size(1)), dtype=label.dtype, device=label.device)
        zero_label = torch.cat((label, zero_fill), dim=0)
        zero_labels.append(zero_label)

    images = torch.stack(images)
    labels = torch.stack(zero_labels)

    return images, labels




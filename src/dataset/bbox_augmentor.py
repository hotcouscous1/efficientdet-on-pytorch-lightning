from src.__init__ import *
import albumentations as A
import cv2
from src.dataset.utils import imagenet_fill



class Bbox_Augmentor:

    __doc__ = r"""
        Examples:        
            T = Bbox_Augmentor(total_prob=0.5, min_area=2048, min_visibility=0.1)

            T.append(A.RandomResizedCrop(256, 512, (0.1, 1.0), p=1))
            T.append(A.HorizontalFlip(p=0.5))
            T.append(A.RandomBrightnessContrast(p=1))
            T.make_compose()

            t_image, t_bboxes, t_category_ids = T(image, bboxes, category_ids).values()

        * if label(bboxes and category_ids) is not given, it returns transformed images and empty lists for label.
    """

    def __init__(self,
                 total_prob: float = 0.5,
                 format: str = 'coco',
                 min_area: float = 0,
                 min_visibility: float = 0,
                 dataset_stat: Optional[Tuple] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ToTensor: bool = True,
                 with_label: bool = True
                 ):

        self.transforms = []
        self.Compose = None

        self.total_prob = total_prob
        self.format = format
        self.min_area = min_area
        self.min_visibility = min_visibility

        self.dataset_stat = dataset_stat
        self.ToTensor = ToTensor
        self.with_label = with_label

        if dataset_stat:
            self.normalizer = A.Normalize(mean=dataset_stat[0], std=dataset_stat[1], max_pixel_value=255.0)


    def append(self, albumentation):
        self.transforms.append(albumentation)


    def remove(self, albumentation):
        self.transforms.remove(albumentation)


    def make_compose(self):
        bbox_params = A.BboxParams(self.format, label_fields=['category_ids'], min_area=self.min_area, min_visibility=self.min_visibility)
        self.Compose = A.Compose(self.transforms, bbox_params=bbox_params, p=self.total_prob)


    def init_compose(self):
        self.Compose = None


    def __call__(self,
                 image: Numpy,
                 bboxes: Optional[List[list]],
                 category_ids: Optional[List[int]]
                 ) -> Dict:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not (bboxes or category_ids):
            bboxes, category_ids = [], []

        if self.Compose:
            output = self.Compose(image=image, bboxes=bboxes, category_ids=category_ids)
        else:
            output = {'image': image, 'bboxes': bboxes, 'category_ids': category_ids}

        if self.dataset_stat:
            output['image'] = self.normalizer(image=output['image'])['image']

        if self.ToTensor:
            output['image'] = np.transpose(output['image'], (2, 0, 1))
            output['image'] = torch.from_numpy(output['image'])

        if not self.with_label:
            if bboxes or category_ids:
                raise ValueError('the label is already given')
            else:
                del output['bboxes']
                del output['category_ids']

        return output




def default_augmentor(
        img_size: Union[int, Tuple[int, int]]
) -> Bbox_Augmentor:

    if isinstance(img_size, tuple):
        h, w = img_size
    else:
        h, w = img_size, img_size

    aug = Bbox_Augmentor(1, 'coco', min_area=512, min_visibility=0.2)

    aug.append(A.HorizontalFlip(p=0.5))
    aug.append(A.RandomResizedCrop(h, w, (0.8, 1.0), p=0.5))
    aug.append(A.RandomScale((-0.2, 0), p=0.5))

    aug.append(A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT, value=imagenet_fill(), p=0.5))
    aug.append(A.Resize(h, w, p=1))
    aug.make_compose()

    return aug


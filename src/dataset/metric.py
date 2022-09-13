import os
import json
from typing import Dict, OrderedDict, Optional
from pycocotools.coco import COCO
from src.dataset.utils import fill_empty_category



def COCO_Metric(
        resultFile: str,
        annFile: str
) -> Dict:

    from pycocotools.cocoeval import COCOeval

    annType = 'bbox'
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resultFile)
    imgIds = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    summary = [round(i.item(), 3) for i in cocoEval.stats]
    summary = {'AP': summary[0],
               'AP50': summary[1],
               'AP75': summary[2],
               'APsmall': summary[3],
               'APmedium': summary[4],
               'APlarge': summary[5],
               'AR1': summary[6],
               'AP10': summary[7],
               'AR': summary[8],
               'ARsmall': summary[9],
               'ARmedium': summary[10],
               'ARlarge': summary[11]}

    return summary



def Evaluate_COCO(
        result: OrderedDict,
        resultFile: str,
        annFile: Optional[str],
        test: bool = False
) -> Dict:

    missing_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
    result_json = []

    for img_id in result:
        preds = result[img_id]
        try:
            img_id = int(img_id)
        except ValueError:
            pass

        for pred in preds:
            pred = pred.tolist()
            bbox = pred[:4]
            score = pred[4]
            category_id = fill_empty_category(int(pred[5]), missing_ids, start_id=1)

            result_json.append({'image_id': img_id, 'category_id': category_id, 'bbox': bbox, 'score': score})


    result_dir = os.path.dirname(resultFile)

    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(resultFile, 'w') as f:
        json.dump(result_json, f)

    if not test:
        return COCO_Metric(resultFile, annFile)
    else:
        print('the result file has been saved.')


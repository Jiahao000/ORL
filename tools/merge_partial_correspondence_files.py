import argparse

import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge generated partial correspondence json files.')
    parser.add_argument('--input', nargs='+', type=str, help='Input json files to merge.')
    parser.add_argument('--output', type=str, help='Output merged json file.')
    parser.add_argument('--num_images', type=int, choices=[118287, 241690],
        help='The total number of training images: 118287 for COCO, 241690 for COCO+.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # dict
    data_new = {}
    pseudo_anno = {}
    data_list = []
    intra_bboxes = []
    inter_bboxes = []
    for json_file in args.input:
        data = mmcv.load(json_file)
        data_list.append(data)
    for data in data_list:
        for bbox in data['pseudo_annotations']['bbox']:
            intra_bboxes.append(bbox)
        for bbox in data['pseudo_annotations']['knn_bbox_pair']:
            inter_bboxes.append(bbox)
    assert len(intra_bboxes) == len(inter_bboxes) == args.num_images, \
        "Mismatch the total number of training images {}, got: intra {} inter: {}".format(
            args.num_images, len(intra_bboxes), len(inter_bboxes))
    # save to dict
    data_new['info'] = data_list[0]['info']
    data_new['images'] = data_list[0]['images']
    pseudo_anno['image_id'] = data_list[0]['pseudo_annotations']['image_id']
    pseudo_anno['bbox'] = intra_bboxes
    pseudo_anno['knn_image_id'] = data_list[0]['pseudo_annotations']['knn_image_id']
    pseudo_anno['knn_bbox_pair'] = inter_bboxes
    data_new['pseudo_annotations'] = pseudo_anno
    mmcv.dump(data_new, args.output)
    print("All json files have been merged to: {}".format(args.output))

if __name__ == '__main__':
    main()

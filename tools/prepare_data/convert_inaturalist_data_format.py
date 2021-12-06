import argparse

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iNaturalist18 json to txt format.')
    parser.add_argument('--input', type=str, help='Input json file.')
    parser.add_argument('--output', type=str, help='Output txt file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = mmcv.load(args.input)
    output_lines = []
    for img_item in data['images']:
        for ann_item in data['annotations']:
            if ann_item['image_id'] == img_item['id']:
                output_lines.append(
                    "{} {}\n".format(
                        img_item['file_name'], ann_item['category_id']
                    )
                )
    assert len(output_lines) == len(data['images'])
    with open(args.output, 'w') as f:
        f.writelines(output_lines)


if __name__ == '__main__':
    main()

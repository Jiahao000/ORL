import argparse

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO json to txt format.')
    parser.add_argument('--input', nargs='+', type=str, help='Input json files.')
    parser.add_argument('--output', type=str, help='Output txt file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_list = []
    fns = []
    for json_file in args.input:
        data = mmcv.load(json_file)
        data_list.append(data)
    for data in data_list:
        for item in data['images']:
            fns.append(item['file_name'])
    output_lines = ["{}\n".format(fn) for fn in fns]
    with open(args.output, 'w') as f:
        f.writelines(output_lines)


if __name__ == '__main__':
    main()

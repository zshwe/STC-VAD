import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=16, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default=r'/root/autodl-tmp/VadClip/model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default=r'/root/autodl-tmp/VadClip/model/checkpoint_ucf.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default=r'/root/autodl-tmp/VadClip/list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default=r'/root/autodl-tmp/VadClip/list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default=r'/root/autodl-tmp/VadClip/list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default=r'/root/autodl-tmp/VadClip/list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default=r'/root/autodl-tmp/VadClip/list/gt_label_ucf.npy')

parser.add_argument('--lr', default=4e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])

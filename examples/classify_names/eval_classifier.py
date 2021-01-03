from fairseq import checkpoint_utils, data, options, tasks

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='simple_classification')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)
# Load model
print('| loading model from {}'.format('checkpoints/checkpoint_best.pt'))
models, _model_args = checkpoint_utils.load_model_ensemble(['checkpoints/checkpoint_best.pt'], task=task)
model = models[0]

#while True:
#sentence = input('\nInput: ')
#chars = ' '.join(list(sentence.strip()))
#tokens = task.source_dictionary.encode_line(
#    chars, add_if_not_exist=False,
#)
chars = 'W i l l i a m'
tokens = task.source_dictionary.encode_line(
    chars, add_if_not_exist=False,
)
chars = 'W i l l i a m'
tokens2 = task.source_dictionary.encode_line(
    chars, add_if_not_exist=False,
)
sample1 = {"id": 0, "source": tokens}
sample2 = {"id": 1, "source": tokens2}
sample_s = [sample1, sample2]
batch = data.language_pair_dataset.collate(
    samples=sample_s,
    pad_idx=task.source_dictionary.pad(),
    eos_idx=task.source_dictionary.eos(),
    left_pad_source=False,
    input_feeding=False,
)
preds = model(**batch['net_input'])

# Print top 3 predictions and their log-probabilities
top_scores, top_labels = preds[0].topk(k=3)
for score, label_idx in zip(top_scores, top_labels):
    label_name = task.target_dictionary.string([label_idx])
    print('({:.2f})\t{}'.format(score, label_name))

top_scores, top_labels = preds[1].topk(k=3)
for score, label_idx in zip(top_scores, top_labels):
    label_name = task.target_dictionary.string([label_idx])
    print('({:.2f})\t{}'.format(score, label_name))

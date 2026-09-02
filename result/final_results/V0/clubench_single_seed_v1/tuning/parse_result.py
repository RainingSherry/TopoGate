#!/usr/bin/env python
import json, sys
payload=json.load(open(sys.argv[1], encoding='utf-8'))
metrics=payload['metrics']
print(metrics['eval_mask_loss'], metrics['latent_view_cosine_mean'], metrics['input_neighbor_overlap'])

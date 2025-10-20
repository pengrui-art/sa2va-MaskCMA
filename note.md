### refcoco_eval
** refcoco 运行命令 **
```
projects/llava_sam2/evaluation/dist_test.sh projects/llava_sam2/evaluation/refcoco_eval.py /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/sa2va_1b_hf_375568 4 --dataset refcoco --split val --work-dir /data1/pengrui/Project/Sa2VA-CMA/work_dirs/eval_results
```

** sa2va-cma-1b refcoco运行结果 **  
============================================ current  
CIoU: 0.8026784658432007, GIoU: 0.8085440993309021 current  
============================================ current  
RES_refcoco_val successfully finished evaluating current  
{'Acc': np.float32(0.80267847), 'CIoU': np.float32(0.80267847), 'mIoU': np.float32(0.8085441)}  

** refcoco+ **
```
projects/llava_sam2/evaluation/dist_test.sh projects/llava_sam2/evaluation/refcoco_eval.py /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/sa2va_1b_hf_375568 4 --dataset refcoco_plus --split val --work-dir /data1/pengrui/Project/Sa2VA-CMA/work_dirs/eval_results
```
** sa2va-cma-1b refcoco+运行结果**  
============================================ current  
CIoU: 0.7490223050117493, GIoU: 0.7602055072784424 current  
============================================ current  
RES_refcoco+_val successfully finished evaluating current  
{'Acc': np.float32(0.7490223), 'CIoU': np.float32(0.7490223), 'mIoU': np.float32(0.7602055)}  

** refcocog **  
```
projects/llava_sam2/evaluation/dist_test.sh projects/llava_sam2/evaluation/refcoco_eval.py /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/sa2va_1b_hf_375568 4 --dataset refcocog --split val --work-dir /data1/pengrui/Project/Sa2VA-CMA/work_dirs/eval_results
```

** refcocog score **  
============================================ current  
CIoU: 0.7695698738098145, GIoU: 0.7741730213165283 current  
============================================ current  
RES_refcocog_val successfully finished evaluating current  
{'Acc': np.float32(0.7695699), 'CIoU': np.float32(0.7695699), 'mIoU': np.float32(0.774173)}  

### Video Segmentation 
#### ReVOS  
```
projects/llava_sam2/evaluation/dist_test.sh /data1/pengrui/Project/MySa2VA/projects/llava_sam2/evaluation/ref_vos_eval.py /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/sa2va_1b_hf_375568 4 --dataset REVOS --work_dir work_dirs/refvos_revos_run1_sa2va-cma
```

```
python /data1/pengrui/CodeSpace/Sa2VA-CMA/tools/eval/eval_revos.py --pred_path /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/refvos_revos_run1_sa2va-cma/results.json --exp_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/meta_expressions_valid_.json --mask_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/revos/mask_dict.json
```

### Mevis
```
projects/llava_sam2/evaluation/dist_test.sh /data1/pengrui/Project/MySa2VA/projects/llava_sam2/evaluation/ref_vos_eval.py /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/sa2va_1b_hf_375568 4 --dataset MEVIS_U --work_dir work_dirs/refvos_mevis_u_run1_sa2va-cma
```

```
python /data1/pengrui/CodeSpace/Sa2VA-CMA/tools/eval/eval_mevis.py /data1/pengrui/CodeSpace/Sa2VA-CMA/work_dirs/refvos_mevis_u_run1_sa2va-cma/results.json --mevis_exp_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/meta_expressions.json --mevis_mask_path /data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/video_datas/mevis/valid_u/mask_dict.json
```
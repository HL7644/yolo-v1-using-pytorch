import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_class=20
images_size=(448,448)

def get_iou(box1, box2): #both boxes in r1,c1,r2,c2 format
  r1=box1[0] #coordinates in row-column perspective
  c1=box1[1]
  r2=box1[2]
  c2=box1[3]
  tr1=box2[0]
  tc1=box2[1]
  tr2=box2[2]
  tc2=box2[3]
  r1_max=max(r1,tr1)
  r2_min=min(r2,tr2)
  c1_max=max(c1,tc1)
  c2_min=min(c2,tc2)
  if r1_max>r2_min or c1_max>c2_min:
    iou=0
  else:
    intersection=(r2_min-r1_max)*(c2_min-c1_max)
    union=(r2-r1)*(c2-c1)+(tr2-tr1)*(tc2-tc1)-intersection
    iou=intersection/union
  return iou

def get_yolo_grids(images_size, feat_h, feat_w):
  #divide image dimension using featmap size
  img_h, img_w=images_size[2],images_size[3]
  grid_h=img_h/feat_h #downsample=64
  grid_w=img_w/feat_w
  grids=torch.zeros(feat_h,feat_w,4).to(device)
  for row in range(feat_h):
    for col in range(feat_w):
      r1=row*grid_h
      r2=r1+grid_h
      c1=col*grid_w
      c2=c1+grid_w
      grids[row,col,:]=torch.FloatTensor([r1,c1,r2,c2]).to(device)
  return grids

def parametrize_bbox(grid_box, bbox):
  #if bbox center is inside the grid cell
  img_h, img_w=images_size[2], images_size[3]
  g_r1=grid_box[0]
  g_c1=grid_box[1]
  g_w=grid_box[3]-grid_box[1]
  g_h=grid_box[2]-grid_box[0]
  bbox_xc=(bbox[1]+bbox[3])/2
  bbox_yc=(bbox[0]+bbox[2])/2
  bbox_w=bbox[3]-bbox[1]
  bbox_h=bbox[2]-bbox[0]
  t_xc=(bbox_xc-g_c1)/g_w #xc,yc: offset w.r.t top left corner (in gridbox scale)
  t_yc=(bbox_yc-g_r1)/g_h
  t_w=bbox_w/img_w
  t_h=bbox_h/img_h
  param_bbox=torch.FloatTensor([t_xc,t_yc,t_w,t_h]).to(device)
  return param_bbox

def get_prediction(grid_box, box_vector):
  img_h, img_w=images_size[2], images_size[3]
  b_xc=box_vector[0]
  b_yc=box_vector[1]
  b_w=box_vector[2]
  b_h=box_vector[3]
  g_r1=grid_box[0] #top left row
  g_c1=grid_box[1] #top left column
  g_w=grid_box[3]-grid_box[1]
  g_h=grid_box[2]-grid_box[0]
  #predictions
  p_xc=b_xc*g_w+g_c1
  p_yc=b_yc*g_h+g_r1
  p_w=b_w*img_w
  p_h=b_h*img_h
  p_r1=p_yc-0.5*p_h
  if p_r1<0:
    p_r1=0
  p_r2=p_yc+0.5*p_h
  if p_r2>img_h:
    p_r2=img_h
  p_c1=p_xc-0.5*p_w
  if p_c1<0:
    p_c1=0
  p_c2=p_xc+0.5*p_w
  if p_c2>img_w:
    p_c2=img_w
  prediction=torch.FloatTensor([p_r1,p_c1,p_r2,p_c2]).to(device)
  return prediction

def get_offset_loss(gt_box_vector, box_vector):
  offset_loss=torch.FloatTensor([0]).to(device)
  offset_loss+=(box_vector[0]-gt_box_vector[0])**2
  offset_loss+=(box_vector[1]-gt_box_vector[1])**2
  offset_loss+=(torch.sqrt(box_vector[2])-torch.sqrt(gt_box_vector[2]))**2
  offset_loss+=(torch.sqrt(box_vector[3])-torch.sqrt(gt_box_vector[3]))**2
  return offset_loss

def nms(predictions, classes): #perform nms for rois in a single image w.r.t obj_score
  thresh=0.3
  post_nms_objects=torch.Tensor([]).to(device)
  post_nms_classes=[]
  #perform nms
  while predictions.size(0)>0:
    top_prediction=predictions[0]
    post_nms_objects=torch.cat((post_nms_objects, top_prediction.unsqueeze(dim=0)), dim=0)
    post_nms_classes.append(classes[0])
    predictions=predictions[1:,:]
    classes=classes[1:,:]
    for idx, box in enumerate(predictions):
      iou=get_iou(box, top_prediction)
      if iou>thresh:
        predictions=torch.cat((predictions[:idx], predictions[(idx+1):]), dim=0)
        classes=torch.cat((classes[:idx], classes[(idx+1):]), dim=0)
  post_nms_classes=torch.LongTensor(post_nms_classes)
  return post_nms_objects, post_nms_classes

def object_appearance(bbox, grid_box):
  #whether the bbox center is inside the grid box
  bbox_xc=(bbox[1]+bbox[3])/2
  bbox_yc=(bbox[0]+bbox[2])/2
  if bbox_yc>grid_box[0] and bbox_yc<grid_box[2] and bbox_xc>grid_box[1] and bbox_xc<grid_box[3]:
    return True
  else:
    return False

def get_loss(bboxes_img, cls_labels_img, grid_box, feature_vector):
  #loss per grid cell, assign one bbox per grid cell
  per_grid_loss=torch.FloatTensor([0]).to(device)
  lambd_coord=5
  lambd_noobj=0.5
  #extract vectors
  box_vectors=torch.zeros(2,5).to(device)
  predictions=torch.zeros(2,4).to(device)
  ious=torch.zeros(2).to(device)
  box_vectors[0,:]=feature_vector[:5]
  predictions[0,:]=get_prediction(grid_box, box_vectors[0,:4])
  box_vectors[1,:]=feature_vector[5:10]
  predictions[1:]=get_prediction(grid_box, box_vectors[1,:4])
  class_vector=feature_vector[10:]
  #assign bbox per grid cell
  for bb_idx, bbox in enumerate(bboxes_img):
    if object_appearance(bbox, grid_box):
      appearance=True
      cls_label=cls_labels_img[bb_idx]
      bbox=bboxes_img[bb_idx]
      break
  #if object appears inside the grid cell
  if appearance:
    #for one object: classif loss
    sfmax_class_vector=F.softmax(class_vector, dim=0)
    gt_class_vector=torch.zeros(N_class).to(device)
    gt_class_vector[cls_label]=1.
    per_grid_loss=per_grid_loss+F.cross_entropy(sfmax_class_vector.unsqueeze(dim=0), gt_class_vector.unsqueeze(dim=0))
    #loc loss for resp predictor
    ious[0]=get_iou(predictions[0,:], grid_box)
    ious[1]=get_iou(predictions[1,:], grid_box)
    resp_idx=torch.argmax(ious, dim=0)
    noobj_idx=1-resp_idx
    box_vector=box_vectors[resp_idx]
    gt_box_vector=parametrize_bbox(grid_box, bbox)
    per_grid_loss=per_grid_loss+lambd_coord*get_offset_loss(gt_box_vector, box_vector)
    #conf loss
    obj_conf=box_vectors[resp_idx,4]
    gt_obj_conf=ious[resp_idx]
    noobj_conf=box_vectors[noobj_idx,4]
    gt_noobj_conf=ious[noobj_idx]
    per_grid_loss=per_grid_loss+(obj_conf-gt_obj_conf)**2
    per_grid_loss=per_grid_loss+lambd_noobj*(noobj_conf-gt_noobj_conf)**2
  else:
    #ordinary loc & conf loss
    #find resp predictor
    ious[0]=get_iou(predictions[0,:], grid_box)
    ious[1]=get_iou(predictions[1,:], grid_box)
    resp_idx=torch.argmax(ious, dim=0)
    noobj_idx=1-resp_idx
    #loc loss
    box_vector=box_vectors[resp_idx]
    #no box for non-appearing grid cells
    gt_box_vector=torch.zeros(4).to(device)
    per_grid_loss=per_grid_loss+get_offset_loss(gt_box_vector, box_vector)
    #conf loss
    obj_conf=box_vectors[resp_idx,4]
    gt_obj_conf=ious[resp_idx]
    noobj_conf=box_vectors[noobj_idx,4]
    gt_noobj_conf=ious[noobj_idx]
    per_grid_loss=per_grid_loss+(obj_conf-gt_obj_conf)**2
    per_grid_loss=per_grid_loss+lambd_noobj*(noobj_conf-gt_noobj_conf)**2
      
  return per_grid_loss
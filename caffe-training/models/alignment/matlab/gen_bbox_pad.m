function bbox_new = gen_bbox_pad(im,bbox,pad,keep)
[l,w,~]=size(im);
if keep==0  
  pad_pixel_l = round((bbox(4)-bbox(2))*pad);
  pad_pixel_w = round((bbox(3)-bbox(1))*pad);
%   bbox_new = bbox;
  bbox_new(1) = max(1,bbox(1)-pad_pixel_w);
  bbox_new(2) = max(1,bbox(2)-pad_pixel_l);
  bbox_new(3) = min(w,bbox(3)+pad_pixel_w);
  bbox_new(4) = min(l,bbox(4)+pad_pixel_l);
%   crop = im(bbox_new(1):bbox_new(3),bbox_new(2):bbox_new(4),:);
else
    %keep aspect ratio
    pad_l = (bbox(4)-bbox(2))*(1+2*pad);
    pad_w = (bbox(3)-bbox(1))*(1+2*pad);
    max_s = max(pad_l,pad_w);
    half_s = floor(max_s/2);
    mid_l = floor((bbox(4)+bbox(2))/2);
    mid_w = floor((bbox(3)+bbox(1))/2);
    bbox_new(1) = max(1,mid_w - half_s);
  bbox_new(2) = max(1,mid_l - half_s);
  bbox_new(3) = min(w,mid_w + half_s);
  bbox_new(4) = min(l,mid_l + half_s);
end
end
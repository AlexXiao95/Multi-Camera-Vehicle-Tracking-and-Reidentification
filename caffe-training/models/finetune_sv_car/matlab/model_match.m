clear,close all
modelDir ='/home/ljyang/work/alignment/3d_models/car_models/obj';
temp = dir(modelDir);
temp = temp(3:end);
len = length(temp);
model_name_3d = cell(len,3);
for i=1:len
    model_name_3d{i,1} = temp(i).name;
end

load('/home/ljyang/work/data/CompCars/raw/data/misc/make_model_name');

%%
clc 
close
clear
model_value = load('model_value.txt');
hope_value = load('hope_Value.txt');
random_value = load('random_value.txt');
pid = load('PID.txt');
p = pid(:,1);
i = pid(:,2);
d = pid(:,3);
%���������⼸�У��Ϳ���ֻ��ʾһ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_value = model_value(1:400,1);
hope_value = hope_value(1:400,1);
random_value = random_value(1:400,1);
p = p(1:400);
i = i(1:400);
d = d(1:400);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I=1:length(hope_value);
%������������һ�־Ϳ���
%%%%%%%%%%%%%%%%��һ������ʾ��һ��%%%%%%%%%%%%%%%%%%%%
% subplot(6,1,1),plot(I,hope_value(:,1));
% subplot(6,1,2),plot(I,model_value(:,1));
% subplot(6,1,3),plot(I,random_value(:,1));
% subplot(6,1,4),plot(I,p(:,1));
% subplot(6,1,5),plot(I,i(:,1));
% subplot(6,1,6),plot(I,d(:,1));
%%%%%%%%%%%%%%%%��һ�ֵ�����ʾ%%%%%%%%%%%%%%%%%%%%%%%
subplot(4,1,1),plot(I,hope_value(:,1));hold on;plot(I,model_value(:,1));hold on;plot(I,random_value(:,1));
subplot(4,1,2),plot(I,p(:,1));
subplot(4,1,3),plot(I,i(:,1));
subplot(4,1,4),plot(I,d(:,1));
function new_brain=ReSizeBrain(mri)
% input- large size mri scan
% output- smaller sized scan, removes all zeros
[x,y,z]=size(mri);
new_brain=zeros(90,120,30);
counterx=0;
firstXflag=0;
for i=1:x                     %deal with X
    if find(mri(i,:,:))
        if firstXflag==0
            firstXflag=i;
        end
        counterx=counterx+1;
    end
end
countery=0;
firstYflag=0;
for i=1:y                     %deal with Y
    if find(mri(:,i,:))
        if firstYflag==0
            firstYflag=i;
        end        
        countery=countery+1;
    end
end
counterz=0;
firstZflag=0;
for i=1:z                     %deal with Z
    if find(mri(:,:,i))
        if firstZflag==0
            firstZflag=i;
        end        
        counterz=counterz+1;
    end
end
% fit new sized brain and return the nonzeros brain 
temp_brain=mri(firstXflag:firstXflag+counterx-1,firstYflag:firstYflag+countery-1 ...
    ,firstZflag:firstZflag+counterz-1);
new_brain(2:1+counterx,2:1+countery,2:1+counterz)=temp_brain;
end
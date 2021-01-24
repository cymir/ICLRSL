function Y = Pre_label(gnd)
%% gnd:label vector£»num_l:number of labeled samples£»Y_label:generated label matrix£»
nClass=length(unique(gnd));
Y=zeros(nClass,length(gnd));    
for i=1:length(gnd)
    for j=1:nClass
        if j==gnd(i)
            Y(j,i)=1;   
        end  
    end
end

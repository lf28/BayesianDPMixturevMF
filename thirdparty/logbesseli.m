 function logb=logbesseli(v,x)
    if v<10
        logb=zeros(size(x));
        logb(x<100)=log(besseli(v,x(x<100)));
        logb(x>=100)=log(besseli(v,x(x>=100),1))+x(x>=100);
    else
        sq=sqrt(x.^2+(v+1)^2);
        logb=sq+(v+1/2)*log(x./(v+1/2+sq))-1/2*log(x/2)+(v+1/2)*log((2*v+3/2)/(2*(v+1)))-1/2*log(2*pi);
    end
end
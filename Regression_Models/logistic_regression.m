function [th]=logistic_regression(x,y,th);
m=size(x,1);
pred=-x*th;
ex=1+e.^pred;
sig=1./ex;
er=sig-y;
error=(1/(2*m))*sum((er).^2);
for i=1:1000
     sig=1./ex;
	r=sig-y;
	temp=(0.001/(m)).*sum(r.*x);
	th=th-temp';
	pred=-x*th;
     ex=1+(e.^pred);
     er=sig-y;
	error=(1/(2*m))*sum((er).^2);
  plot(i,error)
  hold on
	end;
endfunction

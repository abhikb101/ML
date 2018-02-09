function [th]=Linear_regression(x,y,th);
m=size(x,1);
pred=x*th;
error=(1/(2*m))*sum((pred-y).^2);
for i=1:500
	temp=(0.001/(m)).*sum((pred-y).*x);
	disp(temp);
	th=th-temp';
	pred=x*th;
	error=(1/(2*m))*sum((pred-y).^2);
	end;
end;

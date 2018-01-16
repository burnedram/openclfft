function my_fft

freq1 = 10;
sampleRate = 100;
t = 0:1/sampleRate:0.125-1/sampleRate; %0.5 seconds duration
x = (1+cos(2*pi*freq1*t))/2*255 - 127;
t = 0:31;
x = 0:31;

N = 2^nextpow2(length(t))
idx = 0:N-1;
hzx = idx*sampleRate/N;
X = fft(x, N);

subplot(3,2,1)
plot(t,x);
title('Input signal')
subplot(3,2,3)
stem(idx, abs(X))
title('FFT bins')
subplot(3,2,5)
stem(hzx, abs(X))
title('FFT hz')

idxRev = bitrevorder(idx)
xPow = zeros(1, N);
xPow(1:length(x)) = x;
xRev = xPow(idxRev + 1);

v = xRev;
for i = 0:2:15
    tmp        = v(i + 1)   + v(i+1 + 1);
    v(i+1 + 1) = v(i + 1)   - v(i+1 + 1);
    v(i + 1)   = tmp;
end
for i = 2:4:15
    v(i + 1)   = v(i + 1)   * W(N, 0);
    v(i+1 + 1) = v(i+1 + 1) * W(N, 4);
end

for i = 0:4:15
    tmp        = v(i + 1)   + v(i+2 + 1);
    v(i+2 + 1) = v(i + 1)   - v(i+2 + 1);
    v(i + 1)   = tmp;
    tmp        = v(i+1 + 1) + v(i+3 + 1);
    v(i+3 + 1) = v(i+1 + 1) - v(i+3 + 1);
    v(i+1 + 1) = tmp;
end
for i = 4:8:15
    v(i + 1)   = v(i + 1)   * W(N, 0);
    v(i+1 + 1) = v(i+1 + 1) * W(N, 2);
    v(i+2 + 1) = v(i+2 + 1) * W(N, 4);
    v(i+3 + 1) = v(i+3 + 1) * W(N, 6);
end

for i = 0:8:15
    tmp        = v(i + 1)   + v(i+4 + 1);
    v(i+4 + 1) = v(i + 1)   - v(i+4 + 1);
    v(i + 1)   = tmp;
    tmp        = v(i+1 + 1) + v(i+5 + 1);
    v(i+5 + 1) = v(i+1 + 1) - v(i+5 + 1);
    v(i+1 + 1) = tmp;
    tmp        = v(i+2 + 1) + v(i+6 + 1);
    v(i+6 + 1) = v(i+2 + 1) - v(i+6 + 1);
    v(i+2 + 1) = tmp;
    tmp        = v(i+3 + 1) + v(i+7 + 1);
    v(i+7 + 1) = v(i+3 + 1) - v(i+7 + 1);
    v(i+3 + 1) = tmp;
end
for i = 8:16:15
    v(i + 1)   = v(i + 1)   * W(N, 0);
    v(i+1 + 1) = v(i+1 + 1) * W(N, 1);
    v(i+2 + 1) = v(i+2 + 1) * W(N, 2);
    v(i+3 + 1) = v(i+3 + 1) * W(N, 3);
    v(i+4 + 1) = v(i+4 + 1) * W(N, 4);
    v(i+5 + 1) = v(i+5 + 1) * W(N, 5);
    v(i+6 + 1) = v(i+6 + 1) * W(N, 6);
    v(i+7 + 1) = v(i+7 + 1) * W(N, 7);
end

for i = 0:16:15
    tmp         = v(i + 1)   + v(i+8 + 1);
    v(i+8 + 1)  = v(i + 1)   - v(i+8 + 1);
    v(i + 1)    = tmp;
    tmp         = v(i+1 + 1) + v(i+9 + 1);
    v(i+9 + 1)  = v(i+1 + 1) - v(i+9 + 1);
    v(i+1 + 1)  = tmp;
    tmp         = v(i+2 + 1) + v(i+10 + 1);
    v(i+10 + 1) = v(i+2 + 1) - v(i+10 + 1);
    v(i+2 + 1)  = tmp;
    tmp         = v(i+3 + 1) + v(i+11 + 1);
    v(i+11 + 1) = v(i+3 + 1) - v(i+11 + 1);
    v(i+3 + 1)  = tmp;
    tmp         = v(i+4 + 1) + v(i+12 + 1);
    v(i+12 + 1) = v(i+4 + 1) - v(i+12 + 1);
    v(i+4 + 1)  = tmp;
    tmp         = v(i+5 + 1) + v(i+13 + 1);
    v(i+13 + 1) = v(i+5 + 1) - v(i+13 + 1);
    v(i+5 + 1)  = tmp;
    tmp         = v(i+6 + 1) + v(i+14 + 1);
    v(i+14 + 1) = v(i+6 + 1) - v(i+14 + 1);
    v(i+6 + 1)  = tmp;
    tmp         = v(i+7 + 1) + v(i+15 + 1);
    v(i+15 + 1) = v(i+7 + 1) - v(i+15 + 1);
    v(i+7 + 1)  = tmp;
end

subplot(3,2,2)
plot(t,x);
title('Input signal')
subplot(3,2,4)
stem(idx, abs(v))
title('FFT bins')
subplot(3,2,6)
stem(hzx, abs(v))
title('FFT hz')

X.'

end
    
function W = W(N, k)
W = exp(-i*2*pi*k/N);
end
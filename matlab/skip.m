function skip(N, fftSize, twiddleSize)
for i = twiddleSize
    log2N = log2(N);
    log2FftSize = log2(fftSize);
    stage = floor_log2(bitshift(i, -log2FftSize) + 1);
    
    low = bitshift(bitand(i, fftSize-1), log2N-log2FftSize-stage);
    high = bitshift(i, -log2FftSize);
    high = high - (bitshift(1, stage) - 1);
    high = bitshift(high, log2N-stage);
    k = bitor(high, low);
    disp(W(N*2, k))
end
end

function x = floor_log2(x)
x = bitor(x, bitshift(x, -1));
x = bitor(x, bitshift(x, -2));
x = bitor(x, bitshift(x, -4));
x = bitor(x, bitshift(x, -8));
x = bitor(x, bitshift(x, -16));
x = popcount(bitshift(x, -1));
end

function x = popcount(x)
x = sum(dec2bin(x)-'0');
end

function W = W(N, k)
W = exp(-i*2*pi*k/N);
end
function S= downdate_SS(data, S)
    if S.nu <= 1
        nu=0;
        ss = zeros(1, length(data));
    else  
        nu= S.nu - 1;
        ss = S.SS - data;
    end
    S.nu=nu;
    S.SS =ss;
end
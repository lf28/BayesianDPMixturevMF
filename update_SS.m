function S= update_SS(data, S)
    nu= S.nu + 1;
    ss = S.SS + data;
    S.nu=nu;
    S.SS =ss;
end
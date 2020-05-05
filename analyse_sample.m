function [K_s, nmi_s, adjrd_s, purity_s, rd_s, mi_s]=analyse_sample(sample, label, toplot)
    s_c=num2cell(sample,1);
    K_s=cellfun(@(x) length(unique(x)), s_c);
    nmi_s = cellfun(@(x) nmi(x, label), s_c);
    adjrd_s = cellfun(@(x) adjrand(x, label), s_c);
    rd_s = cellfun(@(x) randind(x, label), s_c);
    purity_s = cellfun(@(x) purityFun(x, label), s_c);
    mi_s = cellfun(@(x) mutInfo(x, label), s_c);
    if toplot
    figure;
    plot(K_s);
    figure;
    histogram(K_s);
    figure;
    plot(nmi_s);
    end
end

function prity=purityFun(x, label)
    if any(x==0)
        prity = 0;
    else
        prity= cluster_metrics(label, x);
    end
end
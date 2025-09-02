function [t_new,z]=generate_z(zmax,num,noise_a_percent,N,T,num_outliers,outlier_magnitude_percent,Noise,Outliers)
    %arg:
    % zmax - maximum z value,
    % num - numbers of extremums,
    % noise_a_percent - noise amplitude in % (~2),
    % N - number of points,
    % T - End of the time period,
    % num_outliers - number of outliers,
    % outlier_magnitude_percent - outlier amplitude in % (~15),
    % Noise - =1 with noise,else no noise,
    % Outliers -  =1 with Outliers, else no Outliers
    % no need to plot, plot already included
    z = zmax*rand(1,num);
    %t = [0:T-1];
    t = linspace(0,T,num);
    A = noise_a_percent*zmax/100;
    t_new = linspace(0,T,N);
    z = interp1(t, z, t_new, 'pchip'); %pchip makima
    if Noise == 1
        z = z + A*randn(1,length(z));
    end
    outlier_magnitude = outlier_magnitude_percent*zmax/100;
    outlier_indices = randi(length(t_new), num_outliers, 1);
    if Outliers == 1
        for i = 1:num_outliers
            z(outlier_indices(i)) = z(outlier_indices(i))+outlier_magnitude*(2*rand()-1);
        end
    end
    for i=1:length(z)
        z(i)=ceil(z(i));
        if z(i)>zmax
            z(i) = zmax;
        if z(i)<0
            z(i) = 0;
        end
        end
    end
    figure;
    plot(t_new,z);
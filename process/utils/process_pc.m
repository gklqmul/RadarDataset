Ni = 5;

clear X Y Z
j = 1;
X = [];
Y = [];
Z = [];
S = [];

dX = 0.025;
dY = 0.06;
dZ = 0.1;
dRNG = 0.0633;
dEL = 0.04;
dAZ = 0.01;

for i = (Ni+1)/2:length(radar_data)-(Ni+1)/2+1
    RNG{j} = radar_data(i-(Ni-1)/2).rng;
    EL{j} = radar_data(i-(Ni-1)/2).el;
    AZ{j} = radar_data(i-(Ni-1)/2).az;

    X{j} = radar_data(i-(Ni-1)/2).X;
    Y{j} = radar_data(i-(Ni-1)/2).Y;
    Z{j} = radar_data(i-(Ni-1)/2).Z;
    S{j} = radar_data(i-(Ni-1)/2).snr;
    for k = i-(Ni-1)/2+1:i+(Ni-1)/2
        RNG{j} = [RNG{j}, radar_data(k).rng];
        EL{j} = [EL{j}, radar_data(k).el];
        AZ{j} = [AZ{j}, radar_data(k).az];
        S{j} = [S{j}, radar_data(k).snr];
    end

    % randomise
    RNG{j} = RNG{j} + (dRNG*rand([size(RNG{j})]) - dRNG/2);
    EL{j} = EL{j} + (dEL*rand([size(EL{j})]) - dEL/2);
    AZ{j} = AZ{j} + (dAZ*rand([size(AZ{j})]) - dAZ/2);

    X{j} = RNG{j}.*sin(AZ{j}).*cos(EL{j});
    Y{j} = RNG{j}.*cos(AZ{j}).*cos(EL{j});
    Z{j} = RNG{j}.*sin(EL{j});
    %X{j} = Y;
    j = j+1;
end


%% DBSCAN - clustering
EPSILON = 1;
MINPTS = 30;

for j = 1: length(X)
    XYZ{j} = [X{j}', Y{j}', Z{j}'];
    IDX{j} = dbscan(XYZ{j}, EPSILON, MINPTS);
end

%% Centre of Mass & translate
SELECT_INDEX = 1;
M = [];

% First combine points for all frames
for j = 1:length(XYZ)
    M = [M; XYZ{j}(IDX{j} == SELECT_INDEX,:)];
end

% Calculate mean for all combined frames
M = mean(M, 1);

% Check correct clusters are selected
for j = 1:length(X)
    X{j} = XYZ{j}(:,1);
    Y{j} = XYZ{j}(:,2);
    Z{j} = XYZ{j}(:,3);

    % Check that correct cluster is selected
    com = [];
    for k = 1:max(IDX{j})
        X_body{j} = X{j}(IDX{j} == k);
        Y_body{j} = Y{j}(IDX{j} == k);
        Z_body{j} = Z{j}(IDX{j} == k);

        com(k) = norm(mean([X_body{j}, Y_body{j}, Z_body{j}], 1));
    end
    [~,K(j)] = min(com);
end

% Repeat mean calculation using correct clusters
M = [];
for j = 1:length(XYZ)
    M = [M; XYZ{j}(IDX{j} == K(j),:)];
end
M = mean(M, 1);

% Normalise coordinates around mean and separate body from noise
for j = 1:length(X)
    X{j} = XYZ{j}(:,1);
    Y{j} = XYZ{j}(:,2);
    Z{j} = XYZ{j}(:,3);

    % Normalise to origin
    X_body{j} = X{j}(IDX{j} == K(j));
    Y_body{j} = Y{j}(IDX{j} == K(j));
    Z_body{j} = Z{j}(IDX{j} == K(j));
    
    X_noise{j} = X{j}(IDX{j} ~= K(j));
    Y_noise{j} = Y{j}(IDX{j} ~= K(j));
    Z_noise{j} = Z{j}(IDX{j} ~= K(j));
end

%% Plot
j = 1;
Smag = 10;
figure

% s1 = scatter3(X{j}, Y{j}, Z{j}, S{j}./Smag, '.');
%s1 = scatter3(X_body{j}, X_body{j}, X_body{j}, S{j}(IDX{j}==1)./Smag, 'r.');
s1 = scatter3(X_body{j}, Y_body{j}, Z_body{j}, 'r.');
hold on
sn = scatter3(X_noise{j}, Y_noise{j}, Z_noise{j}, 'k.');

xlabel('X')
ylabel('Y')
zlabel('Z')

xlim([-2 2])
ylim([0,4])
zlim([-2, 2])



sT = title(sprintf("j = %d", j));

pause(2)

for j = 2:length(X)
    s1.XData =X_body{j};
    s1.YData =Y_body{j};
    s1.ZData =Z_body{j};
    %s1.SizeData =S{j}(IDX{j}==1)./Smag;

    sn.XData =X_noise{j};
    sn.YData =Y_noise{j};
    sn.ZData =Z_noise{j};

    sT.String = sprintf("j = %d", j);

    pause(0.25)
end

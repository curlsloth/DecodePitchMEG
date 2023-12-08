% This script was adapted from Hermes, D. J. (2023). 
% Pitch Perception. In The Perceptual Structure of Sound (pp. 381-448). 
% Cham: Springer International Publishing.
% https://doi.org/10.1007/978-3-031-25566-3_8

clear
close all
% InitializePSS()
fprintf('%s\n', mfilename);

% Calculate the helix
n = 120;
nrOfOctaves = 3;
phi = linspace(0, 2*pi*nrOfOctaves, nrOfOctaves*n);
x = zeros(1, nrOfOctaves*n);
y = zeros(1, nrOfOctaves*n);
for k = 1:nrOfOctaves*n
    x(k) = cos(phi(k));
    y(k) = sin(phi(k));
end
z = linspace(0, nrOfOctaves, nrOfOctaves*n);

% Calculate the position of the figure on the screen
figX0 = 270; % horizontal position of figure in pixels from left
figY0 = 120; % vertical position of figure in pixels from bottom
panelWidth = 600; % pixels
panelHeights = 600; % pixels
xLabelHeight = 15; % pixels
yLabelWidth = 15; % pixels
xMargin = 15; % pixels
yMargin = 15; % pixels
[figWidth, figHeight] = SetGcf(figX0, figY0, ...
    panelWidth, panelHeights, ...
    xLabelHeight, yLabelWidth, xMargin, yMargin, ...
    [1,1,1]);
% Calculate the positions of panels within the figure
pos = CalculatePanelPositions(figWidth, figHeight, ...
    panelWidth, panelHeights, yLabelWidth, xLabelHeight, ...
    xMargin, yMargin);

% Plot
panel = 1;
axes('Position', pos(panel,:))
set(gca, 'LineWidth', 0.01, 'FontSize', 14, ...
    'NextPlot', 'add', 'FontWeight', 'bold', 'TickDir', 'out', ...
    'XTick', [], 'YTick', [], 'ZTick', [])
plot(x, y, 'LineWidth', 2, 'Color', 'r')
plot3(x, y, z, 'LineWidth', 2, 'Color', [137, 0, 225] / 255)
plot3([0,0], [0,0], [0,3.5], 'b', 'LineWidth', 2);
view(22.5,17.9)
axis([-1.5 1.5 -1.5 1.5 0 nrOfOctaves], 'off')
notes_all = [" C "; "C#"; " D "; "D#"; ...
    " E "; " F "; "F#"; " G "; "G#"; " A "; "A#"; " B "];
% notes_3 = [' C '; '   '; '   '; '   '; ...
%     ' E '; '   '; '   '; '   '; 'G^#'; '   '; '   '; '   '];
notes_3 = ["   "; "   "; "   "; "   "; "   "; "   "; "   "; "   "; "G#6"; "   "; "   "; "   ";...
           "C7 "; "   "; "   "; "   "; "E7 "; "   "; "   "; "   "; "G#7"; "   "; "   "; "   ";...
           "C8 "; "   "; "   "; "   "; "E8 "; "   "; "   "; "   "; "G#8"; "   "; "   "; "   "; "C9 "];
nN = size(notes_all, 1);
x = cos(2*pi*((0:nrOfOctaves*nN)/nN));
y = sin(2*pi*((0:nrOfOctaves*nN)/nN));
for h = 1:nN
    plot([0.95*x(h) 1.05*x(h)], [0.95*y(h) 1.05*y(h)], ...
        'LineWidth', 2, 'Color', 'black')
    text(x(h), y(h), 0, ...
        notes_all(h,:), 'FontSize', 18, 'FontWeight', 'bold', ...
        'Color', 'k', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end
for h = 1:nrOfOctaves*nN+1
    disp(h)
    plot3([0.95*x(h) 1.05*x(h)], [0.95*y(h) 1.05*y(h)], ... 
        [(h-1)/nN (h-1)/nN], 'LineWidth', 2, 'Color', 'k')
    text(x(h), y(h), (h-1)/nN, notes_3(h), ...
        'FontSize', 18, 'FontWeight', 'bold', ...
        'Color', 'k', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end
xlabel('chroma')
zlabel('height')


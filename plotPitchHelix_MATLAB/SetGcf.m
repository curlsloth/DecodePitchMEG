function [figWidth, figHeight] = SetGcf(figX0, figY0, ...
    panelWidths, panelHeights, xLabelHeights, yLabelWidths, ...
    xMargin, yMargin, color)

% SetGcf calculates the position of the figure
% on the screen based on:
% figX0 horizontal position of figure in pixels from left
% figY0 vertical position of figure in pixels from bottom
% panelWidth in pixels
% panelHeights in pixels
% xLabelHeight in pixels
% yLabelWidth in pixels
% xMargin in pixels
% yMargin in pixels
% If xMargin and yMargin are vectors their value is added 
% to the width and the height of the figure for each 
% panel 
if length(xMargin) == 1
    figWidth = sum(yLabelWidths)+sum(panelWidths+xMargin);
else
    figWidth = sum(yLabelWidths)+sum(panelWidths)+sum(xMargin);
end
if length(yMargin) == 1
    figHeight = sum(xLabelHeights)+sum(panelHeights+yMargin);
else
    figHeight = sum(xLabelHeights)+sum(panelHeights)+sum(yMargin);
end
set(gcf, 'Position', [figX0 figY0 figWidth figHeight], ...
    'PaperPositionMode', 'auto', ...
    'Color', color, 'InvertHardCopy', 'off')


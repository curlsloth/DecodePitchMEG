function pos = CalculatePanelPositions(figWidth, figHeight, ...
    panelWidths, panelHeights, yLabelWidth, xLabelHeight, xMargin, yMargin)

m = length(panelHeights);
n = length(panelWidths);
nrOfPanels = m*n;
xlabelHeights = zeros(1, m);
ylabelWidths = zeros(1, n);
if length(xLabelHeight) == 1
    xlabelHeights(1) = xLabelHeight;
else
    xlabelHeights = xLabelHeight;
end
if length(yLabelWidth) == 1
    ylabelWidths(1) = yLabelWidth;
else
    ylabelWidths = yLabelWidth;
end
pos = zeros(nrOfPanels, 4);
pos(1:m, 1) = yLabelWidth(1)/figWidth;
pos(1:m, 3) = panelWidths(1)/figWidth;
pos(1, 2) = xLabelHeight(1)/figHeight;
pos(1, 4) = panelHeights(1)/figHeight;
for pH = 2:m
    pos(pH, 2) =  pos(pH-1, 2) + ...
        (xlabelHeights(pH)+panelHeights(pH-1)+yMargin)/figHeight;
    pos(pH, 4) =  panelHeights(pH)/figHeight;
end
for pW = 2:n
    r = m*(pW-1)+1;
    rows = r:r+m-1;
    pos(rows, 2) = pos(rows-m, 2);
    pos(rows, 4) = pos(rows-m, 4);
    pos(rows, 1) = pos(rows-m, 1) + ...
        (ylabelWidths(pW)+panelWidths(pW-1)+xMargin)/figWidth;
    pos(rows, 3) = panelWidths(pW)/figWidth;
end

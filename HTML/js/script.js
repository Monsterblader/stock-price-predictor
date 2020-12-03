const resizeViews = {
  active: {
    'border-radius': '0px',
    height: '1080px',
    left: '0px',
    top: '0px',
    width: '1920px',
  },

  box: {
    'background-color': '#fff',
    'border-radius': '5px',
    height: '108px',
    position: 'absolute',
    width: '192px',
  },

  one: {
    left: '236px',
    top: '161px',
  },

  two: {
    left: '453px',
    top: '353px',
  },

  three: {
    left: '642px',
    top: '542px',
  },

  four: {
    left: '832px',
    top: '275px',
  },

  five: {
    left: '1083px',
    top: '117px',
  },

  expand: source => {
    const target = d3.select('.box.' + source);
    const cover = d3.select('.box.' + source + ' .cover');

    target.style('z-index', 10);
    // zoom cover to left corner
    target.classed('active', true)
      .transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('border-radius', '0px')
      .style('left', '0px')
      .style('top', '0px')
      .style('width', '1920px')
      .style('height', '1080px')
      .style('opacity', 1);

    cover.transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('left', '0%')
      .style('top', '0%')
      .style('width', '0px')
      .style('height', '0px')
      .style('font-size', '0px');
  },

  restore: source => {
    d3.select('.active .cover')
      .transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('left', '50%')
      .style('top', '50%')
      .style('height', '108px')
      .style('width', '192px')
      .style('font-size', '24px');

    d3.select('.active')
      .classed('active', false)
      .transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('border-radius', '10px')
      .style('left', resizeViews[source].left)
      .style('top', resizeViews[source].top)
      .style('width', '192px')
      .style('height', '108px')
      .style('opacity', 0.9)
      .style('z-index', 1);
  },

  moveView: (left = 0, top = 0) => {
    d3.select('.map')
      .transition()
      .duration(1000)
      .style('left', left + "px")
      .style('top', top + "px");
  },
}

const getChart = () => {
  const cb = data => {
    // const data = [{
    //     name: "A",
    //     x: 10,
    //   }, {
    //     name: "B",
    //     x: 22,
    //   }, {
    //     name: "C",
    //     x: 33,
    //   }, {
    //     name: "D",
    //     x: 20,
    //   }, {
    //     name: "E",
    //     x: 21,
    //   },
    // ];

    // const data2 = [{
    //     name: "A",
    //     x: 11,
    //   }, {
    //     name: "B",
    //     x: 23,
    //   }, {
    //     name: "C",
    //     x: 34,
    //   }, {
    //     name: "D",
    //     x: 25,
    //   }, {
    //     name: "E",
    //     x: 26,
    //   },
    // ];

    //No.1 define the svg
    const graphWidth = 600,
      graphHeight = 450,
      margin = {
        top: 30,
        right: 10,
        bottom: 30,
        left: 85
      },
      totalWidth = graphWidth + margin.left + margin.right,
      totalHeight = graphHeight + margin.top + margin.bottom,
      svg = d3
        .select("#my_dataviz")
        .append("svg")
        .attr("width", totalWidth)
        .attr("height", totalHeight),

      //No.2 define mainGraph
      mainGraph = svg
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")"),

      //No.3 define axises
      categoriesNames = data.map((d) => d.name),
      xScale = d3
        .scalePoint()
        .domain(categoriesNames)
        .range([0, graphWidth]), // scalepoint make the axis starts with value compared with scaleBand
      yScale = d3
        .scaleLinear()
        .range([graphHeight, 0])
        .domain([0, d3.max(data, (data) => data.x)]), //* If an arrow function is simply returning a single line of code, you can omit the statement brackets and the return keyword

      //No.5 make lines
      line = d3
        .line()
        .x(function (d) {
          return xScale(d.name);
        }) // set the x values for the line generator
        .y(function (d) {
          return yScale(d.x);
        }); // set the y values for the line generator
        // .curve(d3.curveMonotoneX); // apply smoothing to the line

    //No.4 set axises
    mainGraph.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + graphHeight + ")")
      .call(d3.axisBottom(xScale));

    mainGraph.append("g")
      .attr("class", "y axis")
      .call(d3.axisLeft(yScale));

    mainGraph.append("path")
      .datum(data) // 10. Binds data to the line
      .attr("class", "line") // Assign a class for styling
      .attr("d", line); // 11. Calls the line generator

    mainGraph.append("path")
      .datum(data2) // 10. Binds data to the line
      .attr("class", "line") // Assign a class for styling
      .attr("d", line); // 11. Calls the line generator
  } // end cb

  const getParameters = () => {
    const params = {
      ticker: $('#ticker-symbol')[0].value,
      ma50: $('#moving-average-50')[0].checked,
      ma200: $('#moving-average-50')[0].checked,
      RSI: $('#RSI')[0].checked,
      percR: $('#percent-R')[0].checked,
      bolBands: $('#bollinger-bands')[0].checked,
    }

    return params;
  }

  $.ajax({
    url: "getprediction",
    type: "get",
    data: getParameters(),
    success: cb,
  });
} // end getChart

const checkKey = e => {
  const err = e || window.event;
  const getBox = () => {
    let className = '';

    return d3.select('.active')._groups[0][0].classList[1];
  }

  if (e.keyCode == '27') {
    resizeViews.restore(getBox());
  }
}

document.onkeyup = checkKey;

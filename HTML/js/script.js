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
    left: '1114px',
    top: '117px',
  },

  six: {
    left: '976px',
    top: '457px',
  },

  seven: {
    left: '1403px',
    top: '628px',
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
      .style('top', '0px')
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


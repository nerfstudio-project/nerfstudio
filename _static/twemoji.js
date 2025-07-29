function addEvent(element, eventName, fn) {
    if (element.addEventListener)
        element.addEventListener(eventName, fn, false);
    else if (element.attachEvent)
        element.attachEvent('on' + eventName, fn);
}

addEvent(window, 'load', function() {
    twemoji.parse(document.body, {'folder': 'svg', 'ext': '.svg'});
});

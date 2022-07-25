
// this code uses Leva
// instead of dat.GUI

export class WidgetNode extends Component {
    constructor(props) {
        super(props);
        this.state = {
            widget_views: []
        };
    }
}

export class WidgetModel extends Component {
    constructor(props) {
        super(props);
        this.state = {
            guid: null // globally unique identifier
        };
    }
}

export class WidgetView extends Component {
    constructor(props) {
        super(props);
        this.state = {
            model: null, // a WidgetModel this is bound to this view...
            meta_data: {} // data that is passed in from the python side
        };
    }

    // on change, this will broadcast a message to the bridge server
    // then, the bridge server will broadcast a message to update all views (in case multiple windows are open)
}

export class NumberView extends WidgetView {
    constructor(props) {
        super(props);
        this.state = {
            widget_views: []
        };
    }
}

export class SliderView extends WidgetView { }

export class ToggleView extends WidgetView { }

export class IntervalView extends WidgetView { }
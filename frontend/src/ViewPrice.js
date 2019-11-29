import React from 'react';

class ViewPrice extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            result: ''
        }
        this.get_estimate = this.get_estimate.bind(this);
    }

    get_estimate() {
        var data = this.props.data;
        var dictified = {};

        for(var attrib in data) {
            var converted = data[attrib] == "" ? "" : data[attrib] * 1;
            data[attrib] = isNaN(converted) ? data[attrib] : converted;
            var x = attrib;
            dictified[attrib] = data[attrib];
          }
        var jsonified = JSON.stringify(dictified);
        
        
        fetch('http://localhost:3000/api/predict', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: jsonified
        })
            .then(res => res.json())
            .then(res => this.setState({ result: res.prediction }))
    }

    componentDidMount() {
        this.get_estimate();
    }

    render() {
        return (
            <div>
                <h1>${this.state.result}</h1>
            </div>
        )
    }
}

export default ViewPrice;
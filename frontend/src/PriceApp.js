import React from 'react';
import HouseForm from './HouseForm';
import ViewPrice from './ViewPrice'

class PriceApp extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            submitted: false
        };
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleSubmit(e) {
        this.setState(
            {
                submitted: true,
                formValues: e
            })
        console.log(e);
    }


    render() {
        var items = [];
        if(this.state.submitted) {
            items.push(<ViewPrice key="vp" data={this.state.formValues}/>)
        }
        return (
            <div>
                <HouseForm onSubmit={this.handleSubmit} />
                {items}
            </div>

        )
    }
}

export default PriceApp;
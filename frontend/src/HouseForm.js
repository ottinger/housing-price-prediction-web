import React from 'react';
import { updateExpression } from '@babel/types';

let fieldNames = {
	land_size: "Land size",
	year_built: "Year built (average)",
	sqft_sum: "Total sq ft on parcel",
	bed_sum: "Total number of beds",
	bath_sum: "Total number of baths",
	room_sum: "Total number of rooms",
	days_ago: "Days ago",
	main_bldg_year: "Primary building year built",
	main_bldg_effective_year: "Primary building effective year built",
	Sub: "Subdivision name",
	HVAC: "Primary building HVAC type",
	Descr: "Primary building description",
	Roof: "Primary house roof type",
	Exterior: "Primary house exterior type",
	lat: "Latitude",
	lon: "Longitude"
};

class HouseForm extends React.Component {
	constructor(props) {
		super(props);

		this.state = {
			formValues: {
				land_size: '',
				year_built: '',
				sqft_sum: '',
				bed_sum: '',
				bath_sum: '',
				days_ago: '',
				main_bldg_year: '',
				main_bldg_effective_year: '',
				room_sum: '',
				Sub: '',
				HVAC: '',
				Descr: '',
				Exterior: '',
				Roof: '',
				lat: '35.4676',
				lon: '-97.5164'
			},
			submitted: false
		}
		this.handleSubmit = this.handleSubmit.bind(this);
		this.handleChange = this.handleChange.bind(this);
	}

	handleSubmit(e) {
		e.preventDefault();

		this.props.onSubmit(this.state.formValues);
		this.setState({submitted: true});
	}

	handleChange(id,val) {
		var curValues = this.state.formValues;
		curValues[id] = val;
		this.setState({
			formValues: curValues
		});
	}

	render() {
		var items = [];

		if(this.state.submitted) {
			for(var key in fieldNames) {
				items.push(<TextItem key={key} name={key} displayName={fieldNames[key]} 
					default_val={this.state.formValues[key]} />)
			}
			return (
				<div>
					{items}
				</div>
			)
		} else {
			for(var key in fieldNames) {
				items.push(<FormItem key={key} name={key} displayName={fieldNames[key]} 
					default_val={this.state.formValues[key]} onItemChange={this.handleChange} />)
			}
			return (
				<div>
					<form onSubmit={this.handleSubmit}>
						{items}
						<input type="submit" name="submit"></input>
					</form>
				</div>
			)
		}

	}

}

class TextItem extends React.Component {
	constructor(props) {
		super(props);
		var val = props.default_val ? props.default_val : "";
		this.state = {
			name: props.name,
			displayName: props.displayName,
			val: val
		}
	}

	render() {
		return (
			<div>
				{this.state.displayName}:&nbsp;
				{this.state.val}
			</div>
		)
	}
}

class FormItem extends React.Component {
	constructor(props) {
		super(props);
		var val = props.default_val ? props.default_val : "";
		this.state = {
			name: props.name,
			displayName: props.displayName,
			val: val
		}
		this.handleChange = this.handleChange.bind(this);

	}
	handleChange(e) {
		//this.setState({val: e.target.value});

		// we can call "onItemChange" whatever we want
		// this calls the function provided in the onItemChange attribute by parent component
		this.props.onItemChange(this.props.name, e.target.value);
	}

	render() {
		return (
			<div>
				{this.state.displayName}:&nbsp;
				<input type="text" name={this.state.name} value={this.props.default_val} onChange={this.handleChange}></input>
			</div>
		)
	}
}

export default HouseForm;

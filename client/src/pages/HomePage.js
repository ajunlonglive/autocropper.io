
import '../styling/homepage.scss';
import demo from '../images/demo.png';
import before from '../images/before.png';
import after from '../images/after.png';
import {Link} from "react-router-dom";
import ReactBeforeSliderComponent from 'react-before-after-slider-component';
import 'react-before-after-slider-component/dist/build.css';


const HomePage = () => {
    const FIRST_IMAGE = {
        imageUrl: after
      };
      const SECOND_IMAGE = {
        imageUrl: before
      };
    
    return (
    <div>
    <div className='locations'> 
        <ul className='locationslist'>
            <li>A John John Holdings LLC Company</li>
            <li>NEW YORK</li>
            <li>SINGAPORE</li>
            <li>SHENZEN</li>
            <li>TEL AVIV</li>
            <li>LITTLE ROCK</li>
        </ul>
    </div>

    <div className='maincontent'>
        <div className='maincontentitem'> <img src={demo} alt="" className='demoimage'/></div>
        <div className='maincontentitem'> 

        <h3> AutoCropper makes it easy to split a scan of photos into individual photos. See how.</h3>

        <ol className='gradient-list'>
            <li>Upload your scans to our portal.</li>
            <li>Our AI engine will splice your images.</li>
            <li>Download your cropped photos to begin sharing.</li>
        </ol>

        <Link to="/upload"> <button className="button-70">Get Started - Upload Scan Here</button> </Link>
        
        </div>
    </div>

    <br></br>
    <br></br>
    <h2> What is AutoCropper? </h2>
    <p>AutoCropper is the only web tool that will automatically split your scans into cropped images. </p>

    <br />
    <h2> Why should I use it? </h2>
    <p> It's free and no download required, unlike the only competitor. I made this to put on my resume! </p>

    {/* <ReactBeforeSliderComponent
    firstImage={FIRST_IMAGE}
    secondImage={SECOND_IMAGE} /> */}

    </div>);
};

export default HomePage;

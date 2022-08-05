import DragDropFile from './DragDropFile';
import { useState, useEffect } from 'react'


const Upload = () => {
    const [processedImages, setProcessedImages] = useState(null);
    const [imageProcessing, setImageProcessing] = useState(false);

    useEffect(() => {
        setTimeout(function () {
          setImageProcessing(false);
        }, 5000);
      }, [imageProcessing]);

    if (imageProcessing) {
        return (
            <div> 
            <h1>Parsing images...</h1>
                    <section className='output-container'>
                     <div align="center" className='output-container-detail-loading'>
                     <div class="lds-ring"><div></div><div></div><div></div><div></div></div>
                     </div>
                    </section>
            
            </div>
        )
    }
    else if (!processedImages) {
        return (
        <div>  
        <h1>File Uploader</h1>
            <DragDropFile setProcessedImages={setProcessedImages} setImageProcessing={setImageProcessing}/>
        </div>);
    } else {
        
        return (
            <div>
            <h1>Cropped Images</h1>
                    <section className='output-container'>
                     <div className='output-container-detail'>
                        <div className="row">
                            {processedImages.map(item => {
                                    return (
                                        <div className="method-col">
                                            <img src={item} alt="" />
                                            {/* <div className="layer"> </div> */}
                                            <a href={`${item}`} download="hihi.jpg">Click to Download the image</a>

                                        </div>
                                        
                                        )
                                })}
                        </div>


                     </div>
                    </section>

                    <div> Download output </div>
            </div>
        )
    }
};

export default Upload;



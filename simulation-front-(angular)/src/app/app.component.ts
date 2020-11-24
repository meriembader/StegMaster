import {Component, OnInit} from '@angular/core';
import {FormBuilder, FormControl, FormGroup, Validators} from '@angular/forms';
import {markAllAsDirty, requiredFileType, toFormData, toResponseBody, uploadProgress} from './upload-file-validator';
import {HttpClient, HttpHeaders} from '@angular/common/http';
declare var jQuery:any;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',

  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  uploadSelector:string="upload";
  uploadFile:FormGroup;
  progress:number=0;
  loading:boolean=false;
  fraudulent:string="";

  constructor( private http:HttpClient,
               private formBuilder: FormBuilder,){

  }
  ngOnInit(): void {
    this.initForm();
  }

  initForm(){
    this.uploadFile =this.formBuilder.group({
      file1: new FormControl(null, [Validators.required, requiredFileType('csv')]),
      file2: new FormControl(null, [Validators.required, requiredFileType('csv')]),
    });
    this.progress = 0;

  }
  submitUpload() {
    if ( !this.uploadFile.valid ) {
      markAllAsDirty(this.uploadFile);
      return;
    }
    this.loading=true;
    let values:object=this.uploadFile.value;
    this.onUploadFile(values)
    .subscribe(
      (res) => {
        if((res>'0.19')){
          this.fraudulent="fraudulent"
        }else{
          this.fraudulent="not fraudulent"

        }
        console.log(res)
      },
      (err)=>{
        console.log(err);
        this.loading=false;
        this.initForm();
      },
      ()=>{
        this.loading=false;
        this.initForm();
      });
  }
  onUploadFile(values:object){
    return this.http.post("http://localhost:5000/upload", toFormData(values), {
      responseType: 'text',
      reportProgress: true,
    })
  }
  uploadHasError( field: string, error: string ) {
    const control = this.uploadFile.get(field);
    return control.dirty && control.hasError(error);
  }
}


